import numpy as np
import random
from ase.io import read
from amptorch.trainer import AtomsTrainer
from amptorch.ase_utils import AMPtorch
from ase.build import molecule
import os
import shutil
import csv

from evaluate import module_evaluate

# fix random seed
np.random.seed(0)
random.seed(0)

class genetic_algorithm():
    def __init__(self, space, num_generations=2, num_init_population=4, num_best_parents=4, prob_mutation=0.1, verbose=False, log=False):
        self.space = space
        self.best_parents = None
        self.num_generations = num_generations
        self.num_best_parents = num_best_parents
        self.prob_mutation = prob_mutation
        self.num_init_population = num_init_population
        self.verbose = verbose
        self.log = log
        # initial round of population
        self.calculate_population(num_init_population)
    
    def run(self):
        for gen in range(self.num_generations):
            if self.verbose is True: 
                print("Generation {}:".format(gen+1))
            self.rank_population(self.num_best_parents)
            self.crossover()
            self.mutation(self.prob_mutation)
            self.report(verbose=self.verbose, log=self.log)
        return self.report_best_parent, self.report_best_score

    def calculate_population(self, num_init_population):
        sorted_keys = sorted(self.space.keys())
        self.sorted_keys = sorted_keys
        _result_list = np.zeros((num_init_population, len(sorted_keys)))
        _result_list = [[] for _ in range(num_init_population)]
        for i, _key in enumerate(sorted_keys):
            search_space = self.space[_key]
            _rand_gen = np.random.randint(0, high=len(search_space), size=num_init_population)
            _masked_space = [search_space[_] for _ in _rand_gen]
            for j, _ in enumerate(_result_list):
                _.append(_masked_space[j])        
        self.best_parents = _result_list
        # self.best_parents = self._mat2listDict(sorted_keys, _result_mat)
        # initialization step
    
    def rank_population(self, num_best_parents):
        scores = self.calculate_fitness(self.best_parents)
        scores, populations = zip(*sorted(zip(scores, self.best_parents), reverse=True,key=lambda x: x[0]))
        # select best parents
        self.best_parents = [populations[_] for _ in range(num_best_parents)]
    
    def crossover(self, num_crossover_children=4):
        parents = self.best_parents
        children = []
        for i in range(num_crossover_children):
            child =[]
            for j, _key in enumerate(self.sorted_keys):
                parent_col = [_[j] for _ in parents]
                parent_select = np.random.randint(0, len(parent_col))
                child.append(parent_col[parent_select])
            children.append(child)
        self.best_parents.extend(children)
        
    
    def mutation(self, prob_mutation):
        parents = self.best_parents
        for parent in parents:
            for j, _key in enumerate(self.sorted_keys):
                mute = np.random.rand()
                if mute < prob_mutation:
                    search_space = self.space[_key]
                    _rand_gen = np.random.randint(0, high=len(search_space))
                    parent[j] = search_space[_rand_gen]
        self.best_parents = parents 

    def calculate_fitness(self, population):
        input_listDict = self._mat2listDict(self.sorted_keys, population)
        scores = []
        for _parent in input_listDict:
            learning_rate = float(_parent["learning_rate"])
            num_nodes = int(_parent["num_nodes"])
            num_layers = int(_parent["num_layers"])
            fitness = -module_evaluate(learning_rate, num_nodes, num_layers)
            scores.append(fitness)
        return scores
    
    def report(self, verbose, log):
        self.report_best_parent = self.best_parents[0]
        self.report_best_score = self.calculate_fitness(self.best_parents)[0]
        if verbose is True: 
            print("Best parent: {}".format(self.report_best_parent))
            print("Best parent score: {}".format(self.report_best_score))
        if log is True:
            message = [_ for _ in self.report_best_parent]
            message.extend([-self.report_best_score])
            self.write_log(message)
    
    def _mat2listDict(self, keys, mat):
        _list = []
        for _row in mat:
            _tmpdict = {}
            for i_key, j in enumerate(_row):
                _tmpdict[keys[i_key]] = j
            _list.append(_tmpdict)
        return _list

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


os.remove("./ga_log.csv")

space = {"learning_rate": np.logspace(-4, 0, num=10),
    "num_nodes": np.linspace(10, 50, 5, dtype=int), 
    "num_layers": np.linspace(2, 10, 4, dtype=int), 
}

# create an ensemble of ga solvers to get the average and std of solution
ga = genetic_algorithm(space, verbose=True, log=True)
ga.run()

shutil.rmtree("./checkpoints")