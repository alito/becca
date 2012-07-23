"""
benchmark 0.4.0

A suite of worlds to characterize the performance of Becca variants.
Other agents may use this benchmark as well, as long as they have the 
same interface. (See Becca documentation for a detailed specification.)

This benchmark more heavily values breadth than virtuosity. Agents that
can perform a wide variety of tasks moderately well will score better 
than agents that perform a single task optimally and all others very poorly.

In order to facilitate apples-to-apples comparisons between agents, the 
benchmark will be version numbered.

In 60 runs on the benchmark, Becca 0.4.0 averaged a score of 0.292
"""

""" Empirically, running this version of the benchmark multiple times
gives values with a standard deviation of about 0.013. If you want more
accurate estimate of an agent's performance, run it 3 or 5 times 
and take the average. 
Or better yet, to help account for the fact that it is a somewhat non-Gaussian 
process, (it has a short tail on the low side, almost none on the high side)
run it 7 times, throw away the two highest and two lowest scores, 
and average the rest.
"""

from agent.agent import Agent
import numpy as np
import matplotlib.pyplot as plt

from worlds.grid_1D import World as World_grid_1D
from worlds.grid_1D_ms import World as World_grid_1D_ms
from worlds.grid_1D_noise import World as World_grid_1D_noise
from worlds.grid_2D import World as World_grid_2D
from worlds.grid_2D_dc import World as World_grid_2D_dc
from worlds.image_1D import World as World_image_1D
from worlds.image_2D import World as World_image_2D


def main():
    
    """ Tabulate the performance from each world """
    performance = []
    
    world = World_grid_1D()
    performance.append(test(world))
    world = World_grid_1D_ms()
    performance.append(test(world))
    world = World_grid_1D_noise()
    performance.append(test(world))
    world = World_grid_2D()
    performance.append(test(world))
    world = World_grid_2D_dc()
    performance.append(test(world))
    world = World_image_1D()
    performance.append(test(world))
    world = World_image_2D()
    performance.append(test(world))
    
    print "Individual benchmark scores: " , performance
    
    total = 0
    for val in performance:
        total += val
    mean_performance = total / len(performance)
    
    print "Overall benchmark score: ", mean_performance 
    
    """ Block the program, displaying all plots.
    When the plot windows are closed, the program closes.
    """
    plt.show()
    
    
def test(world):
    
    MAX_NUM_FEATURES = 1000
    agent = Agent(world.num_sensors, world.num_primitives, 
                  world.num_actions, MAX_NUM_FEATURES)
    
    """ If configured to do so, the world sets some Becca parameters to 
    modify its behavior. This is a development hack, and should eventually be 
    removed as Becca matures and settles on a good, general purpose
    set of parameters.
    """
    world.set_agent_parameters(agent)
         
    """ Give an initial resting action to kick things off. """
    actions = np.zeros(world.num_actions)
    
    """ Repeat the loop through the duration of the existence of the world."""
    while(world.is_alive()):
        sensors, primitives, reward = world.step(actions)
        actions = agent.step(sensors, primitives, reward)
             
    """ Report the performance of the agent on the world. """
    performance = agent.report_performance(show=False)
    
    return performance

    
if __name__ == '__main__':
    main()
