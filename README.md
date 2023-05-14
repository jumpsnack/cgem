# CGEM: Concept Graph Embedding Models

This repository contains the code for our concept graph embedding models (CGEM). The codebase is implemented using PyTorch and tested on Ubuntu 16.04.4 LTS

## Prerequisite

### `Configure environment`

Install [Anaconda](https://www.anaconda.com/).

Create and activate a virtual environment.

    conda create --name cgem python=3.9
    conda activate cgem

The code is tested with python 3.9, cuda == 10.2, and pytorch == 1.12.1.
Install the required additional packages.

    pip install -r requirements.txt

### `Download dataset`

All datset must be downloaded to a directory '../_dataset' and must follow the below organization.

```bash
├──_dataset/
    ├──cub/
        ├──images/
        ...
├──ConceptGraphEmbedding/
    ├──train.py
    ├──config.py
    ├──dataset/
    ├──models/
```

We refer to [CEM](https://github.com/mateoespinosa/cem)'s repository to download and prepare data.


### `Directories`

- `dataset`: contains scripts to load the datasets used in our experiments
- `models`: contains the implementations of our concept encoder and concept GCN models
