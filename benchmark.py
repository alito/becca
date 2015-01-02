"""
benchmark 0.6.0

A suite of worlds to characterize the performance of BECCA variants.
Other agents may use this benchmark as well, as long as they have the 
same interface. (See BECCA documentation for a detailed specification.)
In order to facilitate apples-to-apples comparisons between agents, the 
benchmark will be version numbered.

Run at the command line as a script with no argmuments:
> python benchmark.py

For N_RUNS = 7, Becca 0.6.0 scored 56
"""
import matplotlib.pyplot as plt
import numpy as np
import tester
from core.agent import Agent
from worlds.grid_1D import World as World_grid_1D
from worlds.grid_1D_ms import World as World_grid_1D_ms
from worlds.grid_1D_noise import World as World_grid_1D_noise
from worlds.grid_2D import World as World_grid_2D
from worlds.grid_2D_dc import World as World_grid_2D_dc
from worlds.image_1D import World as World_image_1D
from worlds.image_2D import World as World_image_2D

def main():
    N_RUNS = 7
    benchmark_lifespan = 1e4
    overall_performance = []
    # Run all the worlds in the benchmark and tabulate their performance
    for i in range(N_RUNS):
        performance = []
        world = World_grid_1D(lifespan=benchmark_lifespan)
        performance.append(tester.test(world, show=False))
        world = World_grid_1D_ms(lifespan=benchmark_lifespan)
        performance.append(tester.test(world, show=False))
        world = World_grid_1D_noise(lifespan=benchmark_lifespan)
        performance.append(tester.test(world, show=False))
        world = World_grid_2D(lifespan=benchmark_lifespan)
        performance.append(tester.test(world, show=False))
        world = World_grid_2D_dc(lifespan=benchmark_lifespan)
        performance.append(tester.test(world, show=False))
        world = World_image_1D(lifespan=benchmark_lifespan)
        performance.append(tester.test(world, show=False))
        world = World_image_2D(lifespan=benchmark_lifespan)
        performance.append(tester.test(world, show=False))

        print "Individual benchmark scores: " , performance
        total = 0
        for val in performance:
            total += val
        mean_performance = total / len(performance)
        overall_performance.append(mean_performance)
        print "Overall benchmark score, ", i , "th run: ", mean_performance 
    print "All overall benchmark scores: ", overall_performance 
    
    # Automatically throw away the 2 highest and 2 lowest values 
    # if you choose N_RUNS to be 7 or more.
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

    # Find the average of what's left
    sum_so_far = 0.
    for indx in range(len(overall_performance)):
        sum_so_far += overall_performance[indx]
    typical_performance = sum_so_far / len(overall_performance)
    print "Typical performance score: ", typical_performance 
    
    # Block the program, displaying all plots.
    # When the plot windows are closed, the program closes.
    plt.show()
    
if __name__ == '__main__':
    main()
