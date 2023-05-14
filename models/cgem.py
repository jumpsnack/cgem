import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange
from einops.layers.torch import Rearrange
from models.base import ABC


class ConceptGraphEmbeddingModels(ABC):
    def __init__(self,
                 dim,
                 backbone,
                 pretrained_backbone=True,
                 training_intervention_prob=0.25,
                 incorrect_intervention=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.dim = dim
        self.backbone = backbone(pretrained_backbone)
        self.backbone = nn.Sequential(
            *(list(self.backbone.children())[:-2])
        )

        self.incorrect_intervention = incorrect_intervention
        self.training_intervention_prob = training_intervention_prob

        self.real_concept_size = self.n_concepts - \
            self.configs.get("num_hidden_concepts", 0)
        self.ones = torch.ones(self.real_concept_size)

        self.add_on = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(512, dim, kernel_size=1,
                          stride=1, padding=0, bias=True),
                nn.BatchNorm2d(dim),
                nn.LeakyReLU(),
                Rearrange('B C H W -> B (H W) C')
            ) for _ in range(self.n_concepts)
        ])

        self.bi_gcn = DeformableBipartiteGCN(dim, self.n_concepts)

        self.c2p = nn.Sequential(
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

        self.c_gcn = ConceptGCN(dim, norm=self.configs.get('norm', True), degree_norm=self.configs.get('degree_norm', True))
        self.head = nn.Linear(dim * self.n_concepts, self.n_tasks)

    def forward(self, x, c=None, y=None, train=False, visualize=False, homo=False, **kwargs):
        B = x.size(0)

        h = self.backbone(x)

        c_emb = torch.stack([m(h) for m in self.add_on], dim=0)

        c_emb, pos, A, ref = self.bi_gcn(c_emb)

        intervention_idxs = None

        if self.intervention_policy is not None:
            intervention_idxs, c_int = self.intervention_policy(
                x=x,
                c=c,
                y=y
            )
            intervention_idxs = self._standardize_indices(
                intervention_idxs, x.size(0))

        c_emb = c_emb.permute(1, 0, 2)
        pred_c = self.c2p(c_emb)
        pred_c = rearrange(pred_c, 'B K 1 -> B K')

        if self.intervention_policy is not None and intervention_idxs is not None:
            intervention_idxs = intervention_idxs[:, :self.real_concept_size]
            new_c = c[intervention_idxs]
            if self.incorrect_intervention:
                new_c = (new_c == 0.0).type_as(c)
            pred_c[:, :self.real_concept_size][intervention_idxs] = new_c
            rel = pred_c
        else:
            rel_pred = pred_c.clone()
            rel = self._after_interventions(
                rel_pred[:, :self.real_concept_size],
                intervention_idxs=intervention_idxs,
                c_true=c,
                train=train
            )
            rel = torch.cat([rel, pred_c[:, self.real_concept_size:]], dim=-1)

        emb = self.c_gcn(c_emb, rel)
        emb = emb.view(B, -1)
        pred_y = self.head(emb)

        return pred_y, pred_c[:, :self.real_concept_size]


class DeformableBipartiteGCN(nn.Module):
    def __init__(self, dim, n_concepts) -> None:
        super().__init__()
        self.dim = dim
        self.n_concepts = n_concepts

        self.dh, self. dw = 5, 5
        self.in_h, self.in_w = 10, 10

        self.to_offs = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.LeakyReLU(),
            nn.Linear(dim, 2 * self.dh * self.dw, bias=False),
            Rearrange('B N (p dh dw) -> B N dh dw p', dh=self.dh, dw=self.dw)
        )

        self.gnn_out = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim)
        )

    def forward(self, c_emb):
        K, B = c_emb.size(0), c_emb.size(1)
        c_emb = rearrange(c_emb, 'K B ... -> (K B) ...')

        dist = self.cos_dist(c_emb, c_emb)
        A = torch.exp(-dist)

        c_nodes, max, A = self.embed_center_node(c_emb, A)

        x_i = ((max % self.in_w) / self.in_w) * 2 - 1
        y_i = (torch.floor(max / self.in_h) / self.in_h) * 2 - 1
        ref = torch.cat([x_i, y_i], dim=-1).view(K, B, 1, 1, 2)

        offs = self.to_offs(c_nodes)
        offs = rearrange(offs, '(K B) 1 ... -> K B ...', K=K)
        pos = (ref + offs).tanh()

        support = rearrange(c_emb, '(K B) (H W) C -> K B C H W',
                            K=K, H=self.in_h, W=self.in_w)  # B 16 10 10
        A = rearrange(A, '(K B) 1 (H W) -> K B H W',
                      K=K, H=self.in_h, W=self.in_w)

        edge_features = []
        node_features = []

        for i in range(self.n_concepts):
            _p = pos[i]
            _s = support[i]
            _a = A[i, :, None, ...]

            _s = F.grid_sample(
                _s, _p, mode='bilinear', align_corners=True
            )
            _a = F.grid_sample(
                _a, _p, mode='bilinear', align_corners=True
            )
            node_features.append(_s)
            edge_features.append(_a)

        edge_features = torch.cat(edge_features, dim=1)  # B 112 d d
        node_features = torch.stack(node_features, dim=1)  # B 112 16 d d

        edge_features = rearrange(edge_features, 'B K H W -> (K B) 1 (H W)')
        node_features = rearrange(node_features, 'B K C H W -> (K B) (H W) C')

        c_emb = c_nodes + self.gcn(c_nodes, node_features)

        c_emb = rearrange(c_emb, '(K B) 1 C -> K B C', B=B)
        edge_features = rearrange(edge_features, '(K B) 1 N -> K B N', B=B)

        return c_emb, pos, edge_features, ref

    def gcn(self, x, x_j):
        x_j = x_j.sum(1, keepdim=True)
        h = torch.cat([x, x_j], dim=-1)
        h = self.gnn_out(h)
        return h

    def cos_dist(self, U, V):
        B = U.size(0)
        norm1 = torch.sqrt(torch.sum(U ** 2, dim=-1))
        norm2 = torch.sqrt(torch.sum(V ** 2, dim=-1))

        dot_product = U @ V.permute(0, 2, 1)

        distance = 1 - dot_product / \
            (norm1.view(B, -1, 1) @ norm2.view(B, 1, -1))
        return distance

    def embed_center_node(self, c_emb, A):

        D = torch.sum(A, dim=-1) - torch.diagonal(A, dim1=1, dim2=2)

        _, idx = torch.max(D, dim=-1, keepdim=True)

        D = D.softmax(-1)

        center_nodes = (D.unsqueeze(-1) * c_emb).sum(1).unsqueeze(1)
        new_A = self.batched_index_select(A, idx)
        return center_nodes, idx, new_A

    def batched_index_select(self, t, idx):
        B, N, C = t.shape
        _, K = idx.shape

        idx_base = torch.arange(0, B, device=idx.device).view(-1, 1) * N
        idx = idx + idx_base
        idx = idx.contiguous().view(-1)

        feature = t.contiguous().view(B * N, -1)
        feature = feature[idx, :]
        feature = feature.view(B, K, C).contiguous()
        return feature


class ConceptGCN(nn.Module):
    def __init__(self, dim, norm=True, degree_norm=True, gamma=1) -> None:
        super().__init__()
        self.gamma = gamma
        self.degree_norm = degree_norm

        self.f_dir = nn.Linear(dim, dim, bias=False)
        self.b_dir = nn.Linear(dim, dim, bias=False)

        self.fc = nn.Linear(dim, dim)
        self.fc_norm = nn.LayerNorm(dim) if norm else nn.Identity()

    def forward(self, c_emb, rel):
        _, N, _ = c_emb.shape
        f_emb = self.f_dir(c_emb)
        b_emb = self.b_dir(c_emb)

        with torch.no_grad():
            psuedo_A = rel
            psuedo_A = psuedo_A.unsqueeze(2) - psuedo_A.unsqueeze(1)
            psuedo_A = (psuedo_A + self.gamma)/(2*self.gamma)

            psuedo_B = 1 - psuedo_A

            psuedo_A = self._minus_self_loop(psuedo_A)
            psuedo_B = self._minus_self_loop(psuedo_B)

            if self.degree_norm:
                psuedo_A = self._get_asymmetric_norm(psuedo_A)
                psuedo_B = self._get_asymmetric_norm(psuedo_B)

        g_emb = self.embed_relation(
            f_emb, psuedo_A) + self.embed_relation(b_emb, psuedo_B)
        c_emb = c_emb + self.fc_norm(self.fc(g_emb))
        return c_emb.contiguous()

    def embed_relation(self, emb, A):
        output = einsum(A, emb, 'b n m, b m d -> b n d')
        return output

    def _minus_self_loop(self, A):
        B, N, _ = A.shape
        eye = torch.eye(N).to(A)

        A = A - eye
        A = torch.clip(A, 0, 1)
        return A

    def _get_asymmetric_norm(self, A):
        B, N, _ = A.shape
        arr = torch.arange(N)

        D = torch.sum(A, dim=-1)
        Dinv = torch.zeros_like(A)
        Dinv[..., arr, arr] = D ** -1
        A = torch.matmul(Dinv, A)
        return A
