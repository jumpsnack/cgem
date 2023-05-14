from dataset.cub import CUB


def load_dataset(name: str, **kwargs):
    if name == 'cub':
        dataset = CUB(**kwargs)
        return dataset, 200, dataset.attr_labels, dataset.imbalance
    else:
        NotImplementedError(f"Dataset {name} is not supported!")
