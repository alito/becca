
import numpy as np

class World(object):
    """ The base class for creating a new world """

    def __init__(self):
        self.timestep = 0
        self.LIFESPAN = 10 ** 4
        self.name = 'abstract base world'
        """ These will likely be overridden in any subclass."""
        self.num_sensors = 0
        self.num_primitives = 0
        self.num_actions = 0
        
        self.MAX_NUM_FEATURES = None
        
    def step(self, action):
        self.timestep += 1
        sensors = np.zeros(self.num_sensors)
        primitives = np.zeros(self.num_primitives)
        reward = 0
        return sensors, primitives, reward
    
    def set_agent_parameters(self, agent):
        pass
    
    def is_alive(self):
        if(self.timestep < self.LIFESPAN):
            return True
        else:
            return False
   
    def log(self):
        pass

    def is_time_to_display(self):
        return False
     
    def announce(self):
        print "Entering", self.name
        
    
    
