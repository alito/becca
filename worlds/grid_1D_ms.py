
from .base_world import World as BaseWorld

import numpy as np

class World(BaseWorld):
    """grid_1D_ms.World
    One-dimensional grid task, multi-step

    In this task, the agent steps forward and backward along a
    line. The fourth position is rewarded (+1/2) and the ninth
    position is punished (-1/2).

    This is intended to be as similar as possible to the 
    one-dimensional grid task, but require multi-step planning for optimal 
    behavior.

    Optimal performance is between 0.25 and 0.3 reward per time step.

    """

    def __init__(self):
                
        super(World, self).__init__()
        
        self.REPORTING_PERIOD = 10 ** 4
        self.LIFESPAN = 2 * 10 ** 4
        self.REWARD_MAGNITUDE = 100.
        self.ENERGY_COST = 0.01
        self.JUMP_FRACTION = 0.10
        self.display_state = False
        self.name = 'multi-step one dimensional grid world'
        self.announce()
        
        self.num_sensors = 0
        self.num_primitives = 9
        self.num_actions = 3

        self.world_state = 0            
        self.simple_state = 0
        
            
    def step(self, action): 
        """ Advance the World by one timestep """

        if action is None:
            action = np.zeros(self.num_actions)

        action = np.round(action)
        action = action.ravel()

        self.timestep += 1 

        """ Occasionally add a perturbation to the action to knock it into 
        a different state. """
        if np.random.random_sample() < self.JUMP_FRACTION:
            action += round(np.random.random_sample() * 6) * \
                    np.round(np.random.random_sample(self.num_actions))
                    
            #print('jumping')
        else:
            #print('not jumping')
            pass
            
        energy = action[0] + action[1]
        
        self.world_state += action[0] - action[1]
        
        """ Ensure that the world state falls between 0 and 9 """
        self.world_state -= self.num_primitives * \
                            np.floor_divide(self.world_state, 
                                            self.num_primitives)
        self.simple_state = int(np.floor(self.world_state))
        
        """ Assign primitives as zeros or ones. 
        Represent the presence or absence of the current position in the bin.
        """
        sensors = np.zeros(self.num_sensors)
        primitives = np.zeros(self.num_primitives)
        primitives[self.simple_state] = 1
        
        """ Assign reward based on the current state """
        reward = primitives[8] * (-self.REWARD_MAGNITUDE)
        reward += primitives[3] * (self.REWARD_MAGNITUDE)
        
        """ Punish actions just a little """
        reward -= energy * self.ENERGY_COST
        reward = np.max(reward, -1)
        
        self.display()

        return sensors, primitives, reward

                    
    def set_agent_parameters(self, agent):
        """ Prevent the agent from forming any groups """
        agent.perceiver.NEW_FEATURE_THRESHOLD = 1.0


    def display(self):
        """ Provide an intuitive display of the current state of the World 
        to the user.
        """
        if (self.display_state):
            state_image = ['.'] * self.num_primitives
            state_image[self.simple_state] = 'O'
            print(''.join(state_image))
            
        if (self.timestep % self.REPORTING_PERIOD) == 0:
            print("world age is %s timesteps " % self.timestep)
