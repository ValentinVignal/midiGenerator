# Midi Generator

## Introduction

This project aim to train different neural networks on midi dataset to
generate music

## Context

This repository contains the code of my Master Thesis for my Master in the School of Computing of the [National University of Singapore](https://www.comp.nus.edu.sg/) ***""Music completion with deep probabilistic models"***

The Report, Abstract and Presentation's slides are available [here](https://github.com/ValentinVignal/Dissertation-Report)

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

# How to use it

## Setup

First, get a midi dataset and save all the song in a folder `Dataset`
The emplacement of this folder has to be at the emplacement `../../../../../storage1/valentin/`. 
To change it, go in the files `src/GlobalVariables/path.py` and modify it.
The midi files have to be in a folder.
The name of the folder is the name of the dataset

## Get the information of the dataset

To get the information of the dataset, run

```
python debug/check_dataset.py <DATANAME>
```

Main options:
- `data` is the name of the dataset name
- `-h` or `--help` plot the options of the file
- `--notes-range` the range of the notes. Default is `0:88`
- `--instruments` the instruments separated by a `,` (ex: `Piano,Trombone`)
- `--bach` for bach dataset (4 voices of piano)
- `--mono` to specify to use mono encoding for monophonic music
- `--no-transpose` don't transpose the songs in C major or A minor

The file will print the number of available songs and the `notes-range` to specify to not loss any data

## Compute the dataset

To extract the tensors from a midi dataset and save them

```
python compute_data.py <DATANAME>
```

The main options are:
- `data` is the name of the dataset
- `-h` or `--help` will print the options of the file
- `--notes-range` to specify the range of the notes to consider
- `--instruments` the instruments separated by a `,` (ex: `Piano,Trombone`)
- `--bach` for bach dataset (4 voices of piano)
- `--mono` to specify to use mono encoding for monophonic music
- `--no-transpose` don't transpose the songs in C major or A minor

## Train and use a model

To train, evaluation, create songs with a model, run

```
python train.py
```

The main options are:
- `-h` or `--help` to print the options of the file
- `-d` or `--data` the name of the dataset
- `--data-test` the name of the test dataset
- `-m` or `--model` to specify the model to use (`modelName,nameParam,nbSteps`)
- `-l` or `--load` to load the id of the model (`name-modelName,nameParam,nbSteps-epochs-id`)
- `--mono` is used to use a monophonic dataset
- `--no-transposed` is used to not use a transpose dataset (to C major or A minor)
- `-e` or `--epochs` number of epochs
- `--gpu` specifies the GPU to use
- `-b` or `--batch` the batch size
- `--lr` specifies the learning rate
- `--no-rpoe` is used to not used the RPoE layer (Recurrent Product of Experts)
- `--validation` the proportion of the data tu use as a validation set
- `--predict-offset` the offset of prediction (use `2` for the band player script)
- `--evaluate` to evaluate the model on the test dataset
- `--generate` do the _generate_ task
- `--generate-fill` do the _fill_ task
- `--redo-generate` do the _redo_ task
- `--noise` value of the noise in the inputs of the training data
- `-n` or `--name` to give a name to the model
- `--use-binary` use the sigmoid loss for a _note\_continue_ for monophonic music

To get the _best results_ from the report, run:

```
python train.py -d DATANAME --data-test DATATESTNAME -m MRRMVAE,1,8 --use-binary --mono --evaluate --generate --generate-fill --redo-generate
```

## Band Player

This file load a trained model and use a it to play with the user in real time

```
python controller.py
```

The main options are:
- `-h` or `--help` to print the options of the file
- `--inst` to specify what instrument sound the user wants
- `--tempo` specify the tempo
- `-l` or `--load` to load the id of the model (`name-modelName,nameParam,nbSteps-epochs-id`)
- `--played-voice` the voice played by the user in the band
- `--inst-mask` list of the mask to choose the voices the user wants to play with
(to play the first voice and with the second and last voice for a model with 5 voices: `[1,1,0,0,1]`)
- `--nb-steps-shown` the number of steps showed on the piano roll plot
