import numpy as np
import time
#import matplotlib.pyplot as plt

# import evaluate function
from read_model import calculate_energy

dist0 = 1.5
angle0 = 120

x0 = [ dist0 , angle0 ]
x_bounds = [ [0,np.Inf] , [0,180] ]

t0 = time.time()


# defining solver function
def positive_span_opt( f , x0 , x_bounds , tol , a0 , da , n_iter ):

    # defining positive spanning set given the dimensions of the initial guess
    positive_set = []
    dim = len(x0)

    # d vectors in alignment with positive axes
    for i in range(dim):

        vector = [ 0 for j in range(dim) ]
        vector[i] = 1
        positive_set.append( vector )
    
    # 1 vector in opposition to the positive axes
    vector = [ -np.sqrt(dim)/dim for j in range(dim) ]
    positive_set.append( vector )

    # initializing values for the loop
    a = a0
    xc = x0
    i = 0
    xc_list = [ x0 ]

    while any( [ a[j] > tol[j] for j in range(dim) ] ):

        # defining list of initial points
        x_list = get_x_list( xc , a , dim , positive_set )

        # evaluating at each point
        y_list = eval_x( f , x_list , x_bounds , dim )
        index = np.argmin( y_list )

        # decising what to do next
        if index == 0:

            a = [ a[j] - da for j in range(dim) ]

        elif i > n_iter:

            a = [ 0 for j in range(dim) ]

        else:

            xc = x_list[index]
        
        xc_list.append( xc )
        i += 1

    return xc , xc_list


# a helper function to get points in an iteration
def get_x_list( xc , a , dim , positive_set ):

    x_list = [ xc ]

    # geting non-center points
    for vector in positive_set:

        xi = [ xc[j] + a[j]*vector[j] for j in range(dim) ]
        x_list.append( xi )

    return x_list


# a helper function to evaluate the points
def eval_x( f , x_list , x_bounds , dim ):

    y_list = []

    # looping through points and evaluating
    for x in x_list:

        x_check = sum( [ x[j] > x_bounds[j][1] for j in range(dim) ] )
        x_check += sum( [ x[j] < x_bounds[j][0] for j in range(dim) ] )

        # checking if x falls out of bounds, setting to infinity if infeasible
        if x_check > 0:
            y_list.append( np.Inf )

        # otherwise, evaluating normally
        else:
            y_list.append( f(x) )

    return y_list


( x_soln , xc_list ) = positive_span_opt( calculate_energy , x0 , x_bounds , [ 0.01 , 0.05 ] , [ 1 , 5 ] , 0.05 , 50 )

t_end = time.time()

for xc in xc_list:
    print( xc )
    print( '\n' )

#fig , ax = plt.subplots()
#ax.plot(  )
#plt.show()

print("solution: {}".format(x_soln))
print("time elapsed: {}".format(t_end - t0))
