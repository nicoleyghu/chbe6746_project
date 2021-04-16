import numpy as np
import random
from ase.io import read
from amptorch.trainer import AtomsTrainer
from amptorch.ase_utils import AMPtorch
from ase.build import molecule
import os
import shutil
import csv
import pickle
import pandas as pd
import copy
from scipy.optimize import minimize

from evaluate import module_evaluate

trials_log = open('neldermead_after_GA_trials.txt', 'a')

def objective(params):
    learning_rate = float(params[0])
    num_nodes = int(params[1])
    num_layers = int(params[2])

    results = module_evaluate(learning_rate, num_nodes, num_layers)

    trials_log.write("{}\t{}\t{}\t{}\n".format(learning_rate, num_nodes, num_layers, results))
    trials_log.flush()

    return results

# Nelder-mead results
x0 = [0.002154434690031882, 45, 2]
minimize(objective, x0, method="Nelder-Mead", options={"maxiter":20})
