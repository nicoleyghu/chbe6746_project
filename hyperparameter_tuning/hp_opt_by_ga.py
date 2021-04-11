import numpy as np
import random
from ase.io import read
from amptorch.trainer import AtomsTrainer
from amptorch.ase_utils import AMPtorch
from ase.build import molecule
import os
import csv

from evaluate import module_evaluate

# fix random seed
np.random.seed(0)
random.seed(0)

class genetic_algorithm():
    def __init__(self, objective, space, num_generations=2, num_best_parents=4, cross_point=0.5, prob_mutation=0.1, verbose=False, log=False):
        self.space = space
        self.objective = objective
        self.parents = None
        self.num_generations = num_generations
        self.num_best_parents = num_best_parents
        self.prob_mutation = prob_mutation
        self.verbose = verbose
        self.log = log
        # initial round of population
        self.calculate_population()
    
    def run(self):
        for gen in range(self.num_generations):
            if self.verbose is True: 
                print("Generation {}:".format(gen+1))
            self.rank_population(self.num_best_parents)
            self.crossover()
            self.mutation(self.prob_mutation)
            self.report(verbose=self.verbose, log=self.log)
        return self.report_best_parent, self.report_best_score

    def calculate_population(self):
        # initialization step
        if self.parents is None:
            if self.bounds is not None: 
                self._get_low_high()
                num_children, num_vars = self.population_size
                self.new_population = [np.random.uniform(low=self.lows[_], high=self.highs[_], size=num_children) for _ in range(num_vars)]
                self.new_population = np.asarray(self.new_population).transpose()
            else:
                self.new_population = np.random.uniform(low=0, high=1, size=self.population_size)
        # crossover and mutate
        # else:
    
    def rank_population(self, num_best_parents):
        scores = self.calculate_fitness(self.new_population)
        scores, populations = zip(*sorted(zip(scores, self.new_population), reverse=True,key=lambda x: x[0]))
        # select best parents
        self.best_parents = populations[0:num_best_parents]
        self.best_parents = np.asarray(self.best_parents)
    
    def crossover(self, num_crossover=2):
        _, num_vars = self.population_size
        parents = self.best_parents

        if num_vars <= 2:
            crosspoint = 0
            for i in range(num_crossover+1):
                mated = [[parents[i][0],  parents[i+1][1]], [parents[i+1][0],  parents[i][1]]]
                parents = np.vstack((parents, np.array(mated)))
            self.best_parents = parents
        else:
            cross_point = random.randint(0, num_vars)
            raise NotImplementedError()
    
    def mutation(self, prob_mutation):
        _, num_vars = self.population_size
        parents = self.best_parents
        shape = parents.shape
        mute = np.random.rand()
        if mute < prob_mutation:
            row = random.randint(0, shape[0]-1)
            col = random.randint(0, shape[1]-1)
            parents[row, col] = np.random.uniform(low=self.lows[col], high=self.highs[col])
        self.new_population = parents 

    def calculate_fitness(self, population):
        images = self._get_images(self.new_population)
        scores = trainer.predict(images)["energy"]
        scores = -np.asarray(scores)
        return scores
    
    def report(self, verbose, log):
        self.report_best_parent = self.best_parents[0]
        self.report_best_score = self.calculate_fitness(self.best_parents)[0]
        if verbose is True: 
            print("Best parent: {}".format(self.report_best_parent))
            print("Best parent score: {}".format(self.report_best_score))
        if log is True:
            message = self.report_best_parent
            self.write_log(message)

    def _get_low_high(self):
        lows = []
        highs = []
        for bound in self.bounds:
            low, high = bound
            lows.append(low)
            highs.append(high)
        self.lows = lows
        self.highs = highs

    def _get_images(self, x0s):
        images = []
        for x0 in x0s:
            OH_bond_length = x0[0] 
            bond_angle = x0[1]
            image = molecule('H2O')
            image.set_distance(0, 2, OH_bond_length, fix=0)
            image.set_angle(1, 0, 2, bond_angle)
            image.set_cell([10, 10, 10])
            image.center()
            images.append(image)
        return images
    
    def write_log(self, message):
        filename = "./ga_log.csv"
        if os.path.exists(filename) is False:
            with open(filename, "w", newline="") as f:
                writer = csv.writer(f, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(message)
        else:
            with open(filename, "a", newline="") as f:
                writer = csv.writer(f, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(message)


# create an ensemble of ga solvers to get the average and std of solution
sols = []
scores = []
for ensemble in range(20):
    ga = genetic_algorithm(trainer, 8, 2, bounds=bounds, num_generations=10, log=False)
    sol, score = ga.run()
    sols.append(sol)
    scores.append(score)
    # break

ensemble_sol = np.mean(np.asarray(sols), axis=0)
ensemble_sol_std = np.std(np.asarray(sols), axis=0)
print(ensemble_sol)
print(ensemble_sol_std)

# build molecule
OH_bond_length = ensemble_sol[0] 
bond_angle = ensemble_sol[1]
image = molecule('H2O')
image.set_distance(0, 2, OH_bond_length, fix=0)
image.set_angle(1, 0, 2, bond_angle)
image.set_cell([10, 10, 10])
image.center()

print(trainer.predict([image])["energy"][0])