import numpy as np
import random
from ase.io import read
from amptorch.trainer import AtomsTrainer
from amptorch.ase_utils import AMPtorch
from ase.build import molecule
import os
import csv

import matplotlib.pyplot as plt
from copy import deepcopy
from evaluate import module_evaluate

# fix random seed
np.random.seed(0)
random.seed(0)

class tabu_algorithm():
    def __init__( self , objective , guess_init , a=0.0001 , short_term_len=2 , max_iter=10 , bounds=None , verbose=False , log=False ):

        self.objective = objective
        self.guess_init = guess_init
        self.a = a

        dtypes = []

        # looping through and typing the data inputs
        for x in guess_init:
            dtypes.append( type(x) )

        self.dtypes = dtypes

        self.short_term_len = short_term_len
        self.max_iter = max_iter

        self.dim = len( guess_init )

        self.bounds = bounds
        self.verbose = verbose
        self.log = log

        # setting up for initial iteration
        self.best_soln = [ guess_init ]
        self.best_objv = [ objective( guess_init ) ]

        self.cand_soln = [ guess_init ]
        self.cand_objv = [ self.best_objv ]

        self.tabu = [ guess_init ]

    def run( self ):

        # looping through iterations
        for i in range( self.max_iter ):

            # getting local points
            local_points = self.get_local_points()
            local_objv = []

            for local_point in local_points:

                # checking for tabu
                if local_point not in self.tabu:

                    in_bounds = 0

                    # checking for bounds
                    for k in range( self.dim ):
                        if local_point[k] < self.bounds[k][0] or local_point[k] > self.bounds[k][1]:
                            in_bounds += 1
                    
                    if in_bounds == 0:
                        objv = self.objective( local_point )
                        local_objv.append( objv )

                        # checking if objective is better than best overall
                        if objv < self.best_objv[-1]:
                            self.best_soln.append( local_point )
                            self.best_objv.append( objv )

                    else:
                        local_objv.append( 10**6 )

                else:
                    local_objv.append( 10**6 )
            
            # determining new best candidate
            ind = np.argmin( local_objv )
            self.cand_soln.append( local_points[ ind ] )
            self.cand_objv.append( local_objv[ ind ] )

            # checking reporting options
            if self.verbose:
                self.report()

            if self.log:
                self.write_log()

            # updating tabu list
            self.tabu.append( self.cand_soln )
            if len( self.tabu ) > self.short_term_len:
                self.tabu.pop(0)

        return self.best_soln , self.best_objv , self.cand_soln , self.cand_objv

    # method to collect local points about the current center, 2 along each axis
    def get_local_points( self ):

        # instantiating list of local points
        local_points = []

        # looping through dimensions of input space to generate 2*N local points
        for j in range( self.dim ):

            # checking for integer
            if self.dtypes[j] is int:
                local_point = deepcopy( self.cand_soln[-1] )
                local_point[j] -= 1
                local_points.append( local_point )

                local_point = deepcopy( self.cand_soln[-1] )
                local_point[j] += 1
                local_points.append( local_point )

            # checking for float
            if self.dtypes[j] is float:
                local_point = deepcopy( self.cand_soln[-1] )
                local_point[j] -= self.a
                local_points.append( local_point )

                local_point = deepcopy( self.cand_soln[-1] )
                local_point[j] += self.a
                local_points.append( local_point )

        return local_points

    # method to report the results of the current iteration
    def report( self ):

        print("Best Overall Solution: {}".format(self.best_soln))
        print("Best Overall Objective: {}".format(self.best_objv))

        print("Best Candidate Solution: {}".format(self.cand_soln))
        print("Best Candidate Objective: {}".format(self.cand_objv))

    # method to write a log describing the algorithm's progress
    def write_log( self ):

        filename = "./ta_log.csv"
        
        # checking if the log already exits
        if os.path.exists( filename ) is False:
            with open( filename , "w" , newline="" ) as f:
                writer = csv.writer( f , delimiter=',' , quotechar='|' , quoting=csv.QUOTE_MINIMAL )
                writer.writerow( self.best_soln )
        
        else:
            with open( filename , "a" , newline="" ) as f:
                writer = csv.writer( f , delimiter=',' , quotechar='|' , quoting=csv.QUOTE_MINIMAL )
                writer.writerow( self.best_soln )

# defining an objective function which works with the above architecture
def objv_func( x ):
    return module_evaluate( x[0] , x[1] , x[2] )

# defining initial guess
x0 = [ 0.001 , 15 , 15 ]

# instantiating the method
ta = tabu_algorithm( objective=objv_func , guess_init=x0 , a=0.0001 , short_term_len=4 , max_iter=20 , bounds=[ [0,1] , [5,10**6] , [2,10**6] ] , verbose=True , log=True )
( best_soln , best_objv , cand_soln , cand_objv ) = ta.run()

#:fig , ax = plt.subplots()

#ax.plot( range(1,len(best_objv)+1) , best_objv )
#ax.plot( range(1,len(cand_objv)+1) , cand_objv )

#fig.show()

# create an ensemble of ga solvers to get the average and std of solution
#sols = []
#scores = []
#for ensemble in range(20):
#    ga = genetic_algorithm(trainer, 8, 2, bounds=bounds, num_generations=10, log=False)
#    sol, score = ga.run()
#    sols.append(sol)
#    scores.append(score)
#    # break

#ensemble_sol = np.mean(np.asarray(sols), axis=0)
#ensemble_sol_std = np.std(np.asarray(sols), axis=0)
#print(ensemble_sol)
#print(ensemble_sol_std)

# build molecule
#OH_bond_length = ensemble_sol[0] 
#bond_angle = ensemble_sol[1]
#image = molecule('H2O')
#image.set_distance(0, 2, OH_bond_length, fix=0)
#image.set_angle(1, 0, 2, bond_angle)
#image.set_cell([10, 10, 10])
#image.center()

#print(trainer.predict([image])["energy"][0])
