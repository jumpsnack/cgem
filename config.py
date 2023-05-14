GLOBAL_CONFIGS = dict(
    seed=42,
    data_root='../_dataset',
    batch_size=256,
    num_workers=8,
    epochs=100,
    optim='sgd'
)

CGEM_CUB_wo_randint = dict(
    name='CGEM_CUB_wo_randint',
    dataset='cub',
    task_loss_weight=1.0,
    concept_loss_weight=5.0,
    dim=16,
    inter_prob=0.,
    lr=0.01
)

CGEM_CUB_randint = dict(
    name='CGEM_CUB_randint',
    dataset='cub',
    task_loss_weight=1.0,
    concept_loss_weight=5.0,
    dim=16,
    inter_prob=0.25,
    lr=0.01
)

CGEM_CUB_wo_randint.update(GLOBAL_CONFIGS)
CGEM_CUB_randint.update(GLOBAL_CONFIGS)
