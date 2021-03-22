import numpy as np
from scipy.optimize import minimize
import time

# import evaluate function
from read_model import calculate_energy

dist0 = 1.5
angle0 = 120

t0 = time.time()
solver = minimize(calculate_energy, np.array([dist0, angle0]), method="nelder-mead",
                options={"xatol":1e-4})
t_end = time.time()

print("solution: {}".format(solver.x))
print("time elapsed: {}".format(t_end - t0))
