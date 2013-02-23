"""
benchmark 0.4.4

A suite of worlds to characterize the performance of BECCA variants.
Other agents may use this benchmark as well, as long as they have the 
same interface. (See BECCA documentation for a detailed specification.)

This benchmark more heavily values breadth than virtuosity. Agents that
can perform a wide variety of tasks moderately well will score better 
than agents that perform a single task optimally and all others very poorly.

In order to facilitate apples-to-apples comparisons between agents, the 
benchmark will be version numbered.

For N_RUNS = 77, Becca 0.4.4 scored 53.1 with a standard deviation of 1.5
"""


import tester
from agent.agent import Agent
from worlds.grid_1D import World as World_grid_1D
from worlds.grid_1D_ms import World as World_grid_1D_ms
from worlds.grid_1D_noise import World as World_grid_1D_noise
from worlds.grid_2D import World as World_grid_2D
from worlds.grid_2D_dc import World as World_grid_2D_dc
from worlds.image_1D import World as World_image_1D
from worlds.image_2D import World as World_image_2D

import matplotlib.pyplot as plt
import numpy as np
import os, sys    

def main():

    '''lib_path = os.path.abspath(os.path.join('..',''))
    sys.path.append(lib_path)
    os.chdir(os.path.join('..',''))

    import tester
    from agent.agent import Agent
    '''
    N_RUNS = 7
    overall_performance = []
    
    for i in range(N_RUNS):
        
        """ Tabulate the performance from each world """
        performance = []
        
        world = World_grid_1D()
        performance.append(tester.test(world, show=False))
        world = World_grid_1D_ms()
        performance.append(tester.test(world, show=False))
        world = World_grid_1D_noise()
        performance.append(tester.test(world, show=False))
        world = World_grid_2D()
        performance.append(tester.test(world, show=False))
        world = World_grid_2D_dc()
        performance.append(tester.test(world, show=False))
        world = World_image_1D()
        performance.append(tester.test(world, show=False))
        world = World_image_2D()
        performance.append(tester.test(world, show=False))
        
        print "Individual benchmark scores: " , performance
        
        total = 0
        for val in performance:
            total += val
        mean_performance = total / len(performance)
        overall_performance.append(mean_performance)
        
        print "Overall benchmark score, ", i , "th run: ", mean_performance 
        
    print "All overall benchmark scores: ", overall_performance 
    
    
    """ Empirically, running version 0.4.0 of the benchmark multiple times
    gave values with a standard deviation of about 5% of the mean. So if you want a more
    accurate estimate of an agent's performance, run it 3 or 5 times 
    and take the average. Or better yet, to help account for the fact 
    that it is a somewhat non-Gaussian process, (it has a short tail 
    on the low side, almost none on the high side)
    run it 7 times, throw away the two highest and two lowest scores, 
    and average the rest. This benchmark will automatically throw away the 
    2 highest and 2 lowest values if you choose N_RUNS to be 7 or more.
    """
    if N_RUNS >= 7:
        for i in range(2):
            highest_val = -10 ** 6
            lowest_val = 10 ** 6
            
            for indx in range(len(overall_performance)):
                if overall_performance[indx] > highest_val:
                    highest_val = overall_performance[indx]
                if overall_performance[indx] < lowest_val:
                    lowest_val = overall_performance[indx]
            overall_performance.remove(highest_val)
            overall_performance.remove(lowest_val)

    """ Find the average of what's left """
    sum_so_far = 0.
    for indx in range(len(overall_performance)):
        sum_so_far += overall_performance[indx]
        
    typical_performance = sum_so_far / len(overall_performance)
    
    print "Typical performance score: ", typical_performance 
    
    """ Block the program, displaying all plots.
    When the plot windows are closed, the program closes.
    """
    plt.show()
    
    
if __name__ == '__main__':
    main()
