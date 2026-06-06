# PGHA-Net

This repository provides the PyTorch implementation of **PGHA-Net**:

**Periodicity-Guided Dynamic Graphs and Hybrid Aggregation Disentangled Network for Traffic Flow Prediction**

## Requirements

The implementation is based on the following environment:

```text
Python 3.9
PyTorch 1.13.1
NumPy
Pandas
PyWavelets
scikit-learn
```

You can install the required packages by running:

```bash
pip install -r requirements.txt
```

A detailed `requirements.txt` will be updated later.

## Datasets

PGHA-Net is evaluated on public traffic forecasting benchmarks covering two types of spatio-temporal prediction scenarios.

For highway traffic flow prediction, we use the PeMS datasets:

```text
PEMS03
PEMS04
PEMS07
PEMS08
```

For urban travel demand prediction, we use the NYC datasets:

```text
NYCBike
NYCTaxi
```

The processed datasets should be placed under the `data/` directory. The expected directory structure is:

```text
data/
├── PEMS03/
├── PEMS04/
├── PEMS07/
├── PEMS08/
├── bike_drop/
├── bike_pick/
├── taxi_drop/
└── taxi_pick/
```

## Train Commands

It is easy to train PGHA-Net. You can customize the dataset, training epochs, and save path in `train.py`.

### PEMS08

```text
train.py --data PEMS08 --epochs 200 --save logs/name
```

### NYCBike Drop-off

```text
train.py --data bike_drop --epochs 200 --save logs/name
```

The training logs and model checkpoints will be saved under the specified path.

## Repository Structure

```text
PGHA-Net/
├── data/
├── logs/
├── model.py
├── train.py
├── test.py
├── util.py
├── ranger21.py
├── requirements.txt
└── README.md
```

## Acknowledgments

We sincerely thank the authors of existing open-source traffic forecasting frameworks for their valuable contributions to the community.
