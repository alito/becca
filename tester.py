"""
A main test harness for a general reinforcement learning agent. 
"""

import numpy

"""  Selects the World that the Agent will be placed in. One of these
lines should be uncommented.
"""
from . import worlds.grid_1D.Grid_1D as World
#import worlds.grid_1D.Grid_1D_ms as world
#import worlds.grid_1D.Grid_1D_noise as world

def main():
    
    world = World()
    
    """ A unique identifying string for the agent, allowing specific 
    saved agents to be recalled. 
    """
    agent_name = "test_agent";
    #from agent.agent import Agent
    # debug against agent_stub
    from agent.agent_stub import AgentStub as Agent
    agent = Agent(agent_name, world.num_sensors, world.num_primitives, 
                  world.num_actions)

    """ If configured to do so, the world sets some Becca parameters to 
    modify its behavior. This is a development hack, and should eventually be 
    removed as Becca matures and settles on a good, general purpose
    set of parameters.
    """
    world.set_agent_parameters(agent)
         
    """ Give an initial resting action to kick things off. """
    actions = numpy.zeros(world.num_actions)

    """ Repeat the loop through the duration of the existence of the world."""
    while(world.timestep < world.LIFESPAN):
        sensors, primitives, reward = world.step(actions)
        actions = agent.step(sensors, primitives, reward)
             
    """ Report the performance of the agent on the world. """
    print('final performance is {0:.3f}'.format(agent.final_performance()))
    
    return 0
    
if __name__ == '__main__':
    main()