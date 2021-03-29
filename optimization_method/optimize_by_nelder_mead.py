import numpy as np
from scipy.optimize import minimize
import time
from ase.build import molecule
from amptorch.ase_utils import AMPtorch
# import evaluate function
from read_model import load_trainer,calculate_energy

dist0 = 1.5
angle0 = 120

checkpoint_path = "../checkpoints/2021-03-28-21-51-55-test"
trainer = load_trainer(checkpoint_path)

def calculate_energy_fx(x0, trainer=trainer):
    return calculate_energy(x0, trainer)

t0 = time.time()
solver = minimize(calculate_energy_fx, np.array([dist0, angle0]), method="nelder-mead",
                options={"xatol":1e-4})
t_end = time.time()

print("solution: {}".format(solver.x))
print("time elapsed: {}".format(t_end - t0))
