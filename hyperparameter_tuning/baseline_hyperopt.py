import numpy as np
from hyperopt import fmin, tpe, hp, Trials
from evaluate import module_evaluate

# search space
space = {"learning_rate": hp.loguniform("learning_rate", 
                                    np.log(1e-4), np.log(0.1)),
    "num_nodes": hp.choice("num_nodes", range(10, 50, 10)), 
    "num_layers": hp.choice("num_layers", range(3, 10, 3)), 
}

trials_log = open('hyperopt_trials.txt', 'a')

def objective(params):
    learning_rate = params["learning_rate"]
    num_nodes = params["num_nodes"]
    num_layers = params["num_layers"]

    results = module_evaluate(learning_rate, num_nodes, num_layers)

    trials_log.write("{}\t{}\t{}\t{}\n".format(learning_rate, num_nodes, num_layers, results))
    trials_log.flush()

    return results

trials = Trials()

best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials)

print()
print('============================')
print(best)
print(trials.best_trial['result']['loss'])
print('============================')

trials_log.close()

