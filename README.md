# Midi Generator

## Introduction

This project aim to train different neural networks on midi dataset to
generate music

## Files

### `bayesian-opt.py`

```
python bayesian-opt.py
```

It is used to find the best hyper parameters to train a model

### `compute_data.py`

```
python compute_data.py
```

It is used to compute the `.mid` files of the dataset and create the
*numpy* arrays used to train a model.

### `generate.py`

```
python generate.py
```

Loads a trained model and generate music from it.

### `train.py`

```
python train.py
```

Creates or loads a model, train it, and also generate music at the end.