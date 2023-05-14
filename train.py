import os

import numpy as np
import pytorch_lightning as pl
import torch
from config import CGEM_CUB_wo_randint
from dataset import load_dataset
from models.callbacks import get_callbacks
from models.cgem import ConceptGraphEmbeddingModels
from torch.utils.data.dataloader import DataLoader
from torchvision.models import resnet34


def main():
    config = CGEM_CUB_wo_randint
    pl.seed_everything(42)

    dataset_name = config.get('dataset', 'cub')

    train_dataset, nrof_classes, concept_names, imbalance = load_dataset(
        dataset_name, root=config.get('data_root', '../_dataset'), train=True)
    test_dataset, _, _, _ = load_dataset(
        dataset_name, root=config.get('data_root', '../_dataset'), train=False)
    train_loader = DataLoader(train_dataset,
                              batch_size=config.get('batch_size', 128),
                              shuffle=True,
                              sampler=None,
                              num_workers=config.get('num_workers', 8),
                              pin_memory=True,
                              drop_last=True,
                              persistent_workers=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=config.get('batch_size', 128),
                             shuffle=False,
                             sampler=None,
                             num_workers=config.get('num_workers', 8),
                             pin_memory=True,
                             drop_last=False,
                             persistent_workers=True)
    n_tasks = nrof_classes
    n_concepts = len(concept_names)

    print("Imbalance:", imbalance)
    print("n_tasks:", n_tasks)
    print("n_concept:", n_concepts)

    epochs = config.get('epochs', 100)
    optim_name = config.get('optim', 'sgd')
    optim_params = {
        'lr': config.get('lr', 0.005)
    }
    ckpt_dir = f'ckpt/{config.get("name", "unknown")}'
    callbacks = get_callbacks(ckpt_dir, config.get("name", "unknown"))

    model = ConceptGraphEmbeddingModels(
        configs=config,
        n_concepts=n_concepts,
        n_tasks=n_tasks,
        optim_name=optim_name,
        optim_params=optim_params,
        task_loss_weight=config.get('task_loss_weight', 1.0),
        concept_loss_weight=config.get('concept_loss_weight', 5.0),
        concept_imbalance=imbalance,
        dim=config.get('dim', 16),
        backbone=resnet34,
        pretrained_backbone=True,
        training_intervention_prob=config.get('inter_prob', 0.),
        incorrect_intervention=False,
    )

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=epochs,
        check_val_every_n_epoch=5,
        callbacks=callbacks,
        num_sanity_val_steps=0,
        log_every_n_steps=10,
        logger=None,
    )
    trainer.fit(model, train_loader, test_loader)


if __name__ == '__main__':
    main()
