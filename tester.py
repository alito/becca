"""
A main test harness for a general reinforcement learning agent. 
"""

import numpy as np

from agent.agent import Agent
from agent import viz_utils
        
"""  Select the World that the Agent will be placed in. 
One of these import lines should be uncommented.
"""
#from worlds.base_world import World
#from worlds.grid_1D import World
#from worlds.grid_1D_ms import World
#from worlds.grid_1D_noise import World
#from worlds.grid_2D import World
#from worlds.grid_2D_dc import World
#from worlds.image_1D import World
#from worlds.image_2D import World

from worlds.watch import World


# TODO: write visualization methods for worlds that represent a set of
# sensors, primitives, and actions in an intuitive manner

def main():
    
    world = World()
    
    """ A unique identifying string for the agent, allowing specific
    saved agents to be recalled. 
    """
    agent_name = "test";
    MAX_NUM_FEATURES = 3000
    agent = Agent(world.num_sensors, world.num_primitives, 
                  world.num_actions, MAX_NUM_FEATURES, agent_name)

    """ If uncommented, try to restore the agent from saved data.
    If commented out, start fresh each time.
    """
    #agent = agent.restore()
    
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
        
        """ If the world has the appropriate method, use it to display the 
        feature set.
        """
        try:
            if world.is_time_to_display():
                print "Trying to display features"
                world.vizualize_feature_set(
                    viz_utils.reduce_feature_set(agent.grouper), save_eps=True)
                viz_utils.force_redraw()
        except AttributeError:
            pass
    
    """ Report the performance of the agent on the world. """
    agent.report_performance()
    agent.show_reward_history()
    
    return

    
if __name__ == '__main__':
    main()