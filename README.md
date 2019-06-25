# Image classification with progressive learning

## Installation

This section presents instructions to install necessary packages using [Anaconda Platform](https://www.anaconda.com/distribution/).

```
conda create -n image python=3.7 -y
conda activate image
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch -y
conda install tqdm -y
conda install matplotlib -y
conda install -c conda-forge overrides -y
conda install -c conda-forge tensorboardX -y
```

## Usage
### Generate similarity vectors

To create similarity vectors, edit property set in `generate_similarity_vectors.py` and run:
```
python generate_similarity_vectors.py
```

### Training

Edit parameter setting in `config.py` and run:
```
python main.py
```

This will print out metrics and generate a confusion matrix in `figures` folder.
