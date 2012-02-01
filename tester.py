"""
Created on Jan 11, 2012

@author: brandon_rohrer

A main test harness for a general reinforcement learning agent. 
"""

""" 
selects the World that the Agent will be placed in 
"""
import worlds.grid_1D
#    import worlds.grid_1D_ms
#    import worlds.grid_1D_noise
#    import worlds.grid_2D
#    import worlds.grid_2D_dc
#    import worlds.image_1D
#
#    import worlds.listen

#    import worlds.morris  # under development
     
def main():
    world= worlds.grid_1D.World()       
    world, agent = world.initialize()
    
    while (world.final_performance() < -1):
        world.step(agent.actions, agent)
        agent.step(world.sensors, world.primitives, world.reward)
        
    print('final performance is {0:.3f}'.format(world.final_performance()))

    
    
if __name__ == '__main__':
    main()
