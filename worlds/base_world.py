
import numpy as np

class World(object):
    """ The base class for creating a new world. Can also be used as 
    a dummy world for debugging. The functionality is contained in:
    
        step()  advances the world by one time step
        
    also required is:
    
        is_alive()  returns True until the world ends
    
    For the time being, a BECCA-configuration method for internal use 
    has been found to be convenient as well.
    
        set_agent_parameters()  configures Becca
            parameters specifically for the task at hand, when necessary    
    """

    def __init__(self):
        """ Constructor """

        self.timestep = 0
        self.LIFESPAN = 10 ** 4
        self.name = 'abstract base world'

        """ These will likely be overridden in any subclass."""
        self.num_sensors = 0
        self.num_primitives = 0
        self.num_actions = 0
        
        self.MAX_NUM_FEATURES = None
        

    def step(self, action):
        """ Advances the World by one timestep.
        Returns a 3-tuple: sensors, primitives and reward
        """
        self.timestep += 1
        
        sensors = np.zeros(self.num_sensors)
        primitives = np.zeros(self.num_primitives)
        reward = 0
        
        return sensors, primitives, reward

    
    def set_agent_parameters(self, agent):
        """ Sets parameters in the Becca agent that are specific to a 
        particular world.
        Strictly speaking, this method violates the minimal interface 
        between the agent and the world (observations, action, and reward). 
        Ideally, it will eventually become obselete. As Becca matures it 
        will be able to handle more tasks without changing its parameters.
        """
        pass
    
    
    def is_alive(self):
        if(self.timestep < self.LIFESPAN):
            return True
        else:
            return False
        
    def announce(self):
        print "Entering", self.name
        
    
    