
import numpy as np
from agent.agent import Agent
#from agent import viz_utils
        
"""  Select the World that the Agent will be placed in. One of these import lines should be uncommented. """
#from worlds.base_world import World
#from worlds.grid_1D import World
#from worlds.grid_1D_ms import World
#from worlds.grid_1D_noise import World
#from worlds.grid_2D import World
#from worlds.grid_2D_dc import World
#from worlds.image_1D import World
#from worlds.image_2D import World

""" If you want to run a world of your own, add the appropriate line here """
#from worlds.hello import World
from becca_world_listen.listen import World

def test(world, restore=False, agent_name="test"):
    """ Run 'world' """
    if world.MAX_NUM_FEATURES is None:
        MAX_NUM_FEATURES = 100
    else:
        MAX_NUM_FEATURES = world.MAX_NUM_FEATURES

    agent = Agent(world.num_sensors, world.num_primitives, 
                  world.num_actions, MAX_NUM_FEATURES, agent_name=agent_name)

    if restore:
        agent = agent.restore()
    
    """ If configured to do so, the world sets some Becca parameters to 
    modify its behavior. This is a development hack, and should eventually be 
    removed as Becca matures and settles on a good, general purpose
    set of parameters.
    """
    world.set_agent_parameters(agent)
         
    actions = np.zeros((world.num_actions,1))
    
    """ Repeat the loop through the duration of the existence of the world """
    while(world.is_alive()):
        sensors, primitives, reward = world.step(actions)
        actions = agent.step(sensors, primitives, reward)
        
        """ If the world has the appropriate method, use it to display the feature set """
        try:
            if world.is_time_to_display():                
                world.vizualize(agent)
        except AttributeError:
            pass
    
    return agent.report_performance()


def profile():
    """ Profile BECCA's performance """
    import cProfile
    import pstats
    cProfile.run('test(World())', 'tester_profile')
    p = pstats.Stats('tester_profile')
    p.strip_dirs().sort_stats('time', 'cum').print_stats(30)
    
    
if __name__ == '__main__':
    profile_flag = False
    if profile_flag:
        profile()
    else:
        test(World())
