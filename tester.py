import cProfile
import numpy as np
import pstats

"""
Run a BECCA agent with a world.

To use this module as a top level script, select the World that the Agent 
will be placed in.
Make sure the appropriate import line is included and uncommented below. 
Run from the command line, e.g. 
> python tester.py
"""

# Worlds from the benchmark
#from worlds.base_world import World
#from worlds.grid_1D import World
#from worlds.grid_1D_ms import World
from worlds.grid_1D_noise import World
#from worlds.grid_2D import World
#from worlds.grid_2D_dc import World
#from worlds.image_1D import World
#from worlds.image_2D import World

# If you want to run a world of your own, add the appropriate line here
#from worlds.hello import World
#from becca_world_listen.listen import World
#from becca_world_whitestripes.whitestripes import World
#from becca_world_find_square.find_square import World
#from becca_world_watch.watch import World
#from becca_world_track.track import World
#from becca_world_chase_ball.chase import World

from core.agent import Agent 

testing_lifespan = 1e8
profiling_lifespan = 1e2

def test(world, restore=False, show=True, agent_name=None):
    """ 
    Run BECCA with world.  
    
    If restore=True, this method loads a saved agent if it can find one.
    Otherwise it creates a new one. It connects the agent and
    the world together and runs them for as long as the 
    world dictates.
    
    To profile BECCA's performance with world, manually set
    profile_flag in the top level script environment to True.
    """
    if agent_name is None:
        agent_name = '_'.join((world.name, 'agent'))
    agent = Agent(world.num_sensors, world.num_actions, 
                  agent_name=agent_name, show=show)
    if restore:
        agent = agent.restore()

    # If configured to do so, the world sets some BECCA parameters to 
    # modify its behavior. This is a development hack, and 
    # should eventually be removed as BECCA matures and settles on 
    # a good, general purpose set of parameters.
    world.set_agent_parameters(agent)
    actions = np.zeros((world.num_actions,1))
    
    # Repeat the loop through the duration of the existence of the world 
    while(world.is_alive()):
        sensors, reward = world.step(actions)
        world.visualize(agent)
        actions = agent.step(sensors, reward)
    return agent.report_performance()

def profile():
    """ Profile BECCA's performance """
    print 'profiling BECCA\'s performance...'
    cProfile.run('test(World(lifespan=profiling_lifespan), restore=True)', 
                 'tester_profile')
    p = pstats.Stats('tester_profile')
    #p.strip_dirs().sort_stats('time', 'cum').print_stats(30)
    p.strip_dirs().sort_stats('time', 'cumulative').print_stats(30)
    
if __name__ == '__main__':
    profile_flag = False
    if profile_flag:
        profile()
    else:
        test(World(lifespan=testing_lifespan), restore=True)
