
from .base_world import World as BaseWorld
import numpy as np

class World(BaseWorld):
    """ grid_1D.World,  One-dimensional grid task
    In this task, the agent steps forward and backward along a nine-position line. The fourth position 
    is rewarded and the ninth position is punished. There is also a slight punishment for effort 
    expended in trying to move, i.e. taking actions. This is intended to be a simple-as-possible task for
    troubleshooting BECCA. See Chapter 4 of the Users Guide for details.
    Optimal performance is a reward of about 90 per time step.
    """

    def __init__(self, lifespan=None):
        super(World, self).__init__()
        
        if lifespan is None:
            self.LIFESPAN = 10 ** 4
        else:
            self.LIFESPAN = lifespan
        self.REPORTING_PERIOD = 10 ** 4
        self.REWARD_MAGNITUDE = 100.
        self.ENERGY_COST = 0.01 * self.REWARD_MAGNITUDE
        self.JUMP_FRACTION = 0.1
        self.display_state = True 
        self.name = 'one dimensional grid world'
        self.announce()

        self.num_sensors = 9
        self.num_actions = 9
        self.MAX_NUM_FEATURES = self.num_sensors + self.num_actions

        self.world_state = 0
        self.simple_state = 0

    
    def step(self, action): 
        if action is None:
            action = np.zeros((self.num_actions,1))
        self.timestep += 1 
        
        step_size = (action[0] + 
                 2 * action[1] + 
                 3 * action[2] + 
                 4 * action[3] - 
                     action[4] - 
                 2 * action[5] - 
                 3 * action[6] - 
                 4 * action[7])
                        
        """ An approximation of metabolic energy """
        energy    = (action[0] + 
                 2 * action[1] + 
                 3 * action[2] + 
                 4 * action[3] + 
                     action[4] + 
                 2 * action[5] + 
                 3 * action[6] + 
                 4 * action[7])

        self.world_state = self.world_state + step_size        

        """ At random intervals, jump to a random position in the world """
        if np.random.random_sample() < self.JUMP_FRACTION:
	    self.world_state = self.num_sensors * np.random.random_sample()

        """ Ensure that the world state falls between 0 and 9 """
        self.world_state -= self.num_sensors * np.floor_divide(self.world_state, self.num_sensors)
        self.simple_state = int(np.floor(self.world_state))
        
        """ Assign basic_feature_input elements as binary. 
        Represent the presence or absence of the current position in the bin.
        """
        sensors = np.zeros(self.num_sensors)
        sensors[self.simple_state] = 1

        """Assign reward based on the current state """
        reward = sensors[8] * (-self.REWARD_MAGNITUDE)
        reward += sensors[3] * ( self.REWARD_MAGNITUDE)
        
        """ Punish actions just a little """
        reward -= energy  * self.ENERGY_COST
        reward = np.maximum(reward, -self.REWARD_MAGNITUDE)
        
        self.display(action)
        return sensors, reward
    
        
    def set_agent_parameters(self, agent):
        """ Prevent the agent from forming any groups """
        #agent.perceiver.NEW_FEATURE_THRESHOLD = 1.0
        agent.reward_min = -100.
        agent.reward_max = 100.
        
        
    def display(self, action):
        if (self.display_state):
            state_image = ['.'] * (self.num_sensors + self.num_actions + 2)
            state_image[self.simple_state] = 'O'
            state_image[self.num_sensors:self.num_sensors + 2] = '||'
            action_index = np.nonzero(action)[0]
            if action_index.size > 0:
                for i in range(action_index.size):
                    state_image[self.num_sensors + 2 + action_index[i]] = 'x'
            print(''.join(state_image))
            
        if (self.timestep % self.REPORTING_PERIOD) == 0:
            print("world age is %s timesteps " % self.timestep)
