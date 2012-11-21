
from .base_world import World as BaseWorld

import numpy as np

class World(BaseWorld):
    """ grid_1D.World
    One-dimensional grid task

    In this task, the agent steps forward and backward along a
    nine-position line. The fourth position is rewarded (+1/2) and the ninth
    position is punished (-1/2). There is also a slight punishment
    for effort expended in trying to move, i.e. taking actions.
    
    This is intended to be a simple-as-possible task for
    troubleshooting BECCA.
    
    The theoretically optimal performance without exploration is 0.5 
    reward per time step.
    In practice, the best performance the algorithm can achieve with the 
    exploration levels given is around 0.35 reward per time step.
    """

    def __init__(self):
                
        super(World, self).__init__()
        
        self.REPORTING_PERIOD = 10 ** 3
        self.LIFESPAN = 2 * 10 ** 4
        self.REWARD_MAGNITUDE = 0.5
        self.ENERGY_COST = 0.01
        self.display_state = False
        self.name = 'one dimensional grid world'
        self.announce()


        self.num_sensors = 0
        self.num_primitives = 9
        self.num_actions = 9

        self.world_state = 0
        self.simple_state = 0

    
    def step(self, action): 
        """ Advance the World by one timestep """

        if action is None:
            action = np.zeros(self.num_actions)
        
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
        
        """ Ensure that the world state falls between 0 and 9 """
        self.world_state -= self.num_primitives * \
                            np.floor_divide(self.world_state, 
                                            self.num_primitives)
        self.simple_state = int(np.floor(self.world_state))
        
        """ Assign basic_feature_input elements as binary. 
        Represent the presence or absence of the current position in the bin.
        """
        sensors = np.zeros(self.num_sensors)
        primitives = np.zeros(self.num_primitives)
        primitives[self.simple_state] = 1

        """Assign reward based on the current state """
        reward = primitives[8] * (-self.REWARD_MAGNITUDE)
        reward += primitives[3] * ( self.REWARD_MAGNITUDE)
        
        """ Punish actions just a little """
        reward -= energy  * self.ENERGY_COST
        reward = np.max(reward, -1)
        
        reward *= 10 
        reward -= 100
        
        self.display(action)
        
        return sensors, primitives, reward
    
        
    def set_agent_parameters(self, agent):
        """ Prevent the agent from forming any groups """
        agent.perceiver.NEW_FEATURE_THRESHOLD = 1.0
        
        
    def display(self, action):
        """ Provide an intuitive display of the current state of the World 
        to the user.
        """
        if (self.display_state):
            
            state_image = ['.'] * (self.num_primitives + self.num_actions + 2)
            state_image[self.simple_state] = 'O'
            state_image[self.num_primitives:self.num_primitives + 2] = '||'
            action_index = np.nonzero(action)[0]
            if action_index.size > 0:
                state_image[self.num_primitives + 2 + action_index[0]] = 'x'
            print(''.join(state_image))
            
        if (self.timestep % self.REPORTING_PERIOD) == 0:
            print("world age is %s timesteps " % self.timestep)

        