import numpy as np
import ase
from ase.io import read
from amptorch.trainer import AtomsTrainer
from amptorch.ase_utils import AMPtorch
import random
import os

def split_train_test(input_filename, train_ratio, save=False, filenames=None):
    random.seed(0)
    images = ase.io.read(input_filename, index=":")
    total_len = len(images)
    total_train_len = int(total_len * train_ratio)
    train_idx = generate_random_idx(total_len, total_train_len)
    test_idx = [_ for _ in range(total_len) if _ not in train_idx]
    training_list = [images[_] for _ in train_idx]
    test_list = [images[_] for _ in test_idx]
    if save is True:
        ase.io.write(filenames[0], training_list)
        ase.io.write(filenames[1], test_list)
    return training_list, test_list\

def generate_random_idx(total_length, sample_length):
    return_list = random.sample(range(0, total_length), sample_length)
    return_list = list(set(return_list))
    while len(return_list) < sample_length:
        return_list.append(random.randint(0, total_length))
        return_list = list(set(return_list))
    assert len(return_list) == sample_length
    return return_list


def load_training_data(train_filename, test_filename):

    training_list = ase.io.read(train_filename, index=":")
    test_list = ase.io.read(test_filename, index=":")

    return training_list, test_list

def module_evaluate(learning_rate, num_nodes, num_layers):

    input_filename = "../data/water_dft.traj"

    # split input if there's no split
    if (os.path.exists("../data/train.traj") is False) or (os.path.exists("../data/test.traj") is False):
        print("Creating train_test split. ")
        train_ratio = 0.9
        training_list, test_list = split_train_test(input_filename, train_ratio, save=True, filenames=["../data/train.traj", "../data/test.traj"])
    else: 
        print("Reading train_test split. ")
        training_list, test_list = load_training_data("../data/train.traj", "../data/test.traj")

    sigmas = np.logspace(np.log10(0.02), np.log10(1.0), num=5)
    MCSHs = {
        "MCSHs": {
            "0": {"groups": [1], "sigmas": sigmas},
            "1": {"groups": [1], "sigmas": sigmas},
            "2": {"groups": [1, 2], "sigmas": sigmas},
            "3": {"groups": [1, 2, 3], "sigmas": sigmas},
            "4": {"groups": [1, 2, 3, 4], "sigmas": sigmas},
            "5": {"groups": [1, 2, 3, 4, 5], "sigmas": sigmas},
            # "6": {"groups": [1, 2, 3, 4, 5, 6, 7], "sigmas": sigmas},
        },
        "atom_gaussians": {
            "H": "../MCSH_potentials/H_pseudodensity_2.g",
            "O": "../MCSH_potentials/O_pseudodensity_4.g",
        },
        "cutoff": 8,
    }


    elements = ["H", "O"]
    config = {
        "model": {"get_forces": True, "num_layers": num_layers, "num_nodes": num_nodes},
        "optim": {
            "device": "cpu",
            "force_coefficient": 0.2,
            "lr": learning_rate,
            "batch_size": 8,
            "epochs": 200,
        },
        "dataset": {
            "raw_data": training_list,
            # "val_split": 0.1,
            "elements": elements,
            "fp_scheme": "gmp",
            "fp_params": MCSHs,
            "save_fps": True,
        },
        "cmd": {
            "debug": False,
            "run_dir": "./",
            "seed": 1,
            "identifier": "test",
            "verbose": True,
            "logger": False,
        },
    }

    trainer = AtomsTrainer(config)
    trainer.train()

    predictions = trainer.predict(test_list)

    true_energies = np.array([image.get_potential_energy() for image in test_list])
    pred_energies = np.array(predictions["energy"])

    mae_result = np.mean(np.abs(true_energies - pred_energies))

    return mae_result