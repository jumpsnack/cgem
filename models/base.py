from typing import Any

import numpy as np
import pytorch_lightning as pl
import sklearn.metrics
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_bin_accuracy(c_pred, y_pred, c_true, y_true):
    c_pred = c_pred.reshape(-1).cpu().detach() > 0.5
    y_probs = y_pred.cpu().detach()
    y_pred = y_probs > 0.5
    c_true = c_true.reshape(-1).cpu().detach()
    y_true = y_true.reshape(-1).cpu().detach()
    c_accuracy = sklearn.metrics.accuracy_score(c_true, c_pred)
    c_auc = sklearn.metrics.roc_auc_score(c_true, c_pred, multi_class='ovo')
    c_f1 = sklearn.metrics.f1_score(c_true, c_pred, average='macro')
    y_accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    y_auc = sklearn.metrics.roc_auc_score(y_true, y_probs)
    y_f1 = sklearn.metrics.f1_score(y_true, y_pred)
    return (c_accuracy, c_auc, c_f1), (y_accuracy, y_auc, y_f1)


def compute_accuracy(
        c_pred,
        y_pred,
        c_true,
        y_true,
):
    if (len(y_pred.shape) < 2) or (y_pred.shape[-1] == 1):
        return compute_bin_accuracy(
            c_pred,
            y_pred,
            c_true,
            y_true,
        )
    c_pred = c_pred.reshape(-1).cpu().detach() > 0.5
    y_probs = torch.nn.Softmax(dim=-1)(y_pred).cpu().detach()
    used_classes = np.unique(y_true.reshape(-1).cpu().detach())
    y_probs = y_probs[:, sorted(list(used_classes))]
    y_pred = y_pred.argmax(dim=-1).cpu().detach()
    c_true = c_true.reshape(-1).cpu().detach()
    y_true = y_true.reshape(-1).cpu().detach()
    c_accuracy = sklearn.metrics.accuracy_score(c_true, c_pred)
    try:
        c_auc = sklearn.metrics.roc_auc_score(
            c_true,
            c_pred,
            multi_class='ovo',
        )
    except:
        c_auc = 0.0
    try:
        c_f1 = sklearn.metrics.f1_score(
            c_true,
            c_pred,
            average='macro',
        )
    except:
        c_f1 = 0
    y_accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    try:
        y_auc = sklearn.metrics.roc_auc_score(
            y_true,
            y_probs,
            multi_class='ovo',
        )
    except:
        y_auc = 0.0
    try:
        y_f1 = sklearn.metrics.f1_score(y_true, y_pred, average='macro')
    except:
        y_f1 = 0.0
    return (c_accuracy, c_auc, c_f1), (y_accuracy, y_auc, y_f1)


class ABC(pl.LightningModule):
    def __init__(self,
                 configs: dict,
                 n_concepts: int,
                 n_tasks: int,
                 optim_name: str,
                 optim_params: dict,
                 task_loss_weight: float = 1.0,
                 concept_loss_weight: float = 1.0,
                 concept_imbalance: torch.Tensor = None) -> None:
        super().__init__()

        self.configs = configs
        self.n_concepts = n_concepts
        self.n_tasks = n_tasks

        #### Optimizer ####
        self.optim_name = optim_name
        self.lr = optim_params.get('lr', 1e-3)
        self.weight_decay = optim_params.get('weight_decay', 4e-05)
        self.momentum = optim_params.get('momentum', 0.9)

        #### Loss ####
        self.concept_loss = nn.BCELoss(weight=concept_imbalance)
        self.task_loss = nn.CrossEntropyLoss()
        self.concept_loss_weight = concept_loss_weight
        self.task_loss_weight = task_loss_weight

        #### Log ####
        self.max_avg_c_y_acc = 0
        self.max_c_acc = 0
        self.max_y_acc = 0

        self.intervention_policy = None

    def training_step(self, batch, batch_idx):
        loss, result = self._run_step(batch, batch_idx, train=True)
        self.log_dict(result, prog_bar=True, sync_dist=False)
        return {
            'loss': loss,
            'log': result
        }

    def validation_step(self, batch, batch_idx):
        loss, result = self._run_step(batch, batch_idx, train=False)
        result = {'val/' + key: val for key, val in result.items()}
        self.log_dict(result, prog_bar=False, sync_dist=True)
        return result

    def test_step(self, batch, batch_idx):
        loss, result = self._run_step(batch, batch_idx, train=False)
        result = {'test/' + key: val for key, val in result.items()}
        for name, val in result.items():
            self.log(name, val, prog_bar=True)
        return result

    def validation_epoch_end(self, result):
        if self.global_rank == 0:
            avg_c_y_acc = sum([dic['val/avg_c_y_acc'].mean()
                              for dic in result]) / len(result)
            y_acc = sum([dic['val/y_acc'].mean()
                        for dic in result]) / len(result)
            c_acc = sum([dic['val/c_acc'].mean()
                        for dic in result]) / len(result)
            print(
                f'\n  Test stats: Top-1 {y_acc * 100:.2f}%  c_acc {c_acc * 100:.2f}%  avc_c_y_acc {avg_c_y_acc * 100:.2f}%')
            if self.max_avg_c_y_acc < avg_c_y_acc:
                self.max_avg_c_y_acc = avg_c_y_acc
                self.max_y_acc = y_acc
                self.max_c_acc = c_acc
            print(
                f'  ** Max accuracy: {self.max_y_acc * 100:.2f}%\
                c_accuracy: {self.max_c_acc * 100:.2f}%\
                (avg_c_y_acc: {self.max_avg_c_y_acc * 100:.2f})\n'
            )

    def _run_step(self, batch, batch_idx, train=False, **kwargs):
        if len(batch) == 3:
            x, true_y, true_c = batch
        elif len(batch) == 2:
            x, (true_y, true_c) = batch
        else:
            raise NotImplementedError()

        pred_y, pred_c = self(
            x, c=true_c if train else None, train=train, **kwargs)

        concept_loss = self.concept_loss_weight * \
            self.concept_loss(pred_c, true_c)
        task_loss = self.task_loss_weight * self.task_loss(pred_y, true_y)

        loss = task_loss + concept_loss

        (c_accuracy, c_auc, c_f1), (y_accuracy, y_auc, y_f1) = compute_accuracy(
            pred_c,
            pred_y,
            true_c,
            true_y,
        )

        return loss, {
            "lr": self.optim.param_groups[0]['lr'],
            "c_acc": c_accuracy,
            "y_acc": y_accuracy,
            "c_loss": concept_loss.item(),
            "t_loss": task_loss.item(),
            "t+c_loss": loss.item(),
            "avg_c_y_acc": (c_accuracy + y_accuracy) / 2,
        }

    def _valid_step(self, data, data_idx, intervention_idxs=None):
        x, true_y, true_c = data
        x = x.to(self.device)
        true_c = true_c.to(self.device)
        true_y = true_y.to(self.device)

        pred_y, pred_c = self(x, c=true_c, y=true_y, train=False)

        (c_accuracy, c_auc, c_f1), (y_accuracy, y_auc, y_f1) = compute_accuracy(
            pred_c,
            pred_y,
            true_c,
            true_y,
        )

        return (c_accuracy, c_auc, c_f1), (y_accuracy, y_auc, y_f1)

    def forward(self, x, c=None, train=False, **kwargs):
        raise NotImplementedError

    def _after_interventions(self,
                             prob,
                             intervention_idxs=None,
                             c_true=None,
                             train=False):
        """
        The author of this module is Mateo Espinosa Zarlenga.
        """
        if train and (self.training_intervention_prob != 0) and (
                (c_true is not None) and
                (intervention_idxs is None)
        ):
            # Then we will probabilistically intervene in some concepts
            mask = torch.bernoulli(
                self.ones * self.training_intervention_prob,
            )
            intervention_idxs = torch.tile(
                mask,
                (c_true.shape[0], 1),
            )
        if (c_true is None) or (intervention_idxs is None):
            return prob
        intervention_idxs = intervention_idxs.to(prob.device)
        intervention_idxs = intervention_idxs.to(dtype=torch.int32)
        return prob * (1 - intervention_idxs) + intervention_idxs * c_true

    def _standardize_indices(self, intervention_idxs, batch_size):
        if isinstance(intervention_idxs, list):
            intervention_idxs = np.array(intervention_idxs)
        if isinstance(intervention_idxs, np.ndarray):
            intervention_idxs = torch.IntTensor(intervention_idxs)

        if intervention_idxs is None or (
                isinstance(intervention_idxs, torch.Tensor) and
                ((len(intervention_idxs) == 0)
                 or intervention_idxs.shape[-1] == 0)
        ):
            return None
        if not isinstance(intervention_idxs, torch.Tensor):
            raise ValueError(
                f'Unsupported intervention indices {intervention_idxs}'
            )
        if len(intervention_idxs.shape) == 1:
            # Then we will assume that we will do use the same
            # intervention indices for the entire batch!
            intervention_idxs = torch.tile(
                torch.unsqueeze(intervention_idxs, 0),
                (batch_size, 1),
            )
        elif len(intervention_idxs.shape) == 2:
            assert intervention_idxs.shape[0] == batch_size, (
                f'Expected intervention indices to have batch size {batch_size} '
                f'but got intervention indices with shape {intervention_idxs.shape}.'
            )
        else:
            raise ValueError(
                f'Intervention indices should have 1 or 2 dimensions. Instead we got '
                f'indices with shape {intervention_idxs.shape}.'
            )
        if intervention_idxs.shape[-1] == self.n_concepts:
            # We still need to check the corner case here where all indices are
            # given...
            elems = torch.unique(intervention_idxs)
            if len(elems) == 1:
                is_binary = (0 in elems) or (1 in elems)
            elif len(elems) == 2:
                is_binary = (0 in elems) and (1 in elems)
            else:
                is_binary = False
        else:
            is_binary = False
        if not is_binary:
            # Then this is an array of indices rather than a binary array!
            intervention_idxs = intervention_idxs.to(dtype=torch.long)
            result = torch.zeros(
                (batch_size, self.n_concepts),
                dtype=torch.bool,
                device=intervention_idxs.device,
            )
            result[:, intervention_idxs] = 1
            intervention_idxs = result
        assert intervention_idxs.shape[-1] == self.n_concepts, (
            f'Unsupported intervention indices with shape {intervention_idxs.shape}.'
        )
        if isinstance(intervention_idxs, np.ndarray):
            # Time to make it into a torch Tensor!
            intervention_idxs = torch.BoolTensor(intervention_idxs)
        intervention_idxs = intervention_idxs.to(dtype=torch.bool)
        return intervention_idxs

    def configure_optimizers(self):
        if self.optim_name.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        else:
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )

        self.optim = optimizer
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "t+c_loss",
        }
