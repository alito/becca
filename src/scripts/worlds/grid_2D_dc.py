import numpy as np

from worlds.grid_2D import World as Grid_2D_World

class World(Grid_2D_World):
    """ Two-dimensional grid task
    
    DC stands for decoupled. It's just like the task_grid_2D task 
    except that the two dimensions are decoupled. The basic 
    feature vector represents a row and a column separately, 
    not coupled together.
    In this task, the agent steps North, South, East, or West 
    in a 5 x 5 grid-world.  Position (4,4) is rewarded and (2,2) 
    is punished.  There is also a lesser penalty of for each horizontal
    or vertical step taken. Horizonal and vertical positions are 
    reported separately as basic features, rather than raw sensory inputs. 
    See Chapter 4 of the Users Guide for details.
    Optimal performance is a reward of between 90 per time step.
    """
    def __init__(self, lifespan=None):
        """ Set up the world """    
        Grid_2D_World.__init__(self, lifespan)
        self.name = 'grid_2D_dc'
        self.name_long = 'decoupled two dimensional grid world'
        print "--decoupled"
        self.num_sensors = self.world_size * 2
        self.VISUALIZE_PERIOD = 10 ** 3
        self.display_state = False
            
    def assign_sensors(self):
        """ Construct the sensor array from the state information """
        sensors = np.zeros(self.num_sensors)
        sensors[self.simple_state[0]] = 1
        sensors[self.simple_state[1] + self.world_size] = 1
        return sensors
