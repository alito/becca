import numpy as np

class World(object):
    """ The base class for creating a new world """
    def __init__(self, lifespan=None):
        """ Initialize a new world with some benign default values """
        if lifespan is None:
            self.LIFESPAN = 10 ** 4
        else:
            self.LIFESPAN = lifespan
        self.timestep = 0
        self.name = 'abstract base world'
        # These will likely be overridden in any subclass
        self.num_sensors = 0
        self.num_actions = 0
        
    def step(self, action):
        """ Take a timestep through an empty world that does nothing """
        self.timestep += 1
        sensors = np.zeros(self.num_sensors)
        reward = 0
        return sensors, reward
    
    def is_alive(self):
        """ Returns True when the world has come to an end """
        if(self.timestep < self.LIFESPAN):
            return True
        else:
            return False
   
    def visualize(self, agent):
        """ Let the world show BECCA's internal state as well as its own"""
        print self.timestep, 'timesteps'
