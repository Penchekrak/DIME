# DIME — DIsconnected Minima Ensemble

Project for the course "Bayesian Methods of Machine Learning" at Skoltech

## Source paper

[Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs](https://arxiv.org/abs/1802.10026)

## Project proposal

[Original project proposal](https://github.com/Penchekrak/DIME/blob/master/BayesML%20Proposal.pdf)

## Goals of the project

* Implement an experiment environment in PyTorch Lightning with a small CNN and CIFAR10 dataset
* Implement a convenient interface for running Torch nets with arbitrary state_dicts (needed running nets along the curves in the parameter space)
* Implement the curve fitting algorithm to find curves between local minima
* Find several local minima and low-loss curves between them
* Design approaches for finding minima that cannot be connected by low-loss curves, study their feasibility
* Implement beforementioned approaches and compare results with random independent minima
* Compare performance of ensembles of models with different local minima

## Team members

* [Andrew Spiridonov](https://github.com/Penchekrak)
* [Maksim Nekrashevich](https://github.com/max-nekrashevich)
* [Kirill Tyshchuk](https://github.com/Reason239)

## Experiments reproduction

The experiment workflow is implemented via PyTorch Lightning, which ensures convenient reproducibility. 

The main script `train.py` can be launched with an argument `--config-name=<desired config name>`, where `<desired config name>` can be any .yaml file from the `configs/` folder or a custom config.

All of the model checkpoints necessary for the experiments are provided in the `ckpt/` folder.

The script automatically finds and tries to use a GPU if it is available.