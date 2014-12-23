"""
Decoupled two-dimensional grid task

This is just like the regular 2D grid task, except that it the rows
and columns are sensed separately. This makes the task challenging.
Both the row number and column number need to be taken into 
account in order to know what actions to take. This task requires building
basic sensory data into more complex features in order to do well.
"""
import numpy as np
from grid_2D import World as Grid_2D_World

class World(Grid_2D_World):
    """ Decoupled two-dimensional grid world
    
    It's just like the grid_2D world except that the sensors
    array represents a row and a column separately, 
    rather than coupled together.
    Optimal performance is a reward of about 90 per time step.
    """
    def __init__(self, lifespan=None):
        Grid_2D_World.__init__(self, lifespan)
        self.name = 'grid_2D_dc'
        self.name_long = 'decoupled two dimensional grid world'
        print ", decoupled"
        self.num_sensors = self.world_size * 2
        self.VISUALIZE_PERIOD = 10 ** 3
        self.display_state = False
            
    def assign_sensors(self):
        sensors = np.zeros(self.num_sensors)
        sensors[self.world_state[0]] = 1
        sensors[self.world_state[1] + self.world_size] = 1
        return sensors
