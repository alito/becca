
from .base_world import World as BaseWorld
import numpy as np

class World(BaseWorld):
    """ grid_2D.World, Two-dimensional grid task
    In this task, the agent steps North, South, East, or West in a 5 x 5 grid-world. Position (4,4) 
    is rewarded and (2,2) is punished. There is also a lesser penalty for each horizontal or vertical step 
    taken. Horizonal and vertical positions are reported separately as basic features, rather than 
    raw sensory inputs.See Chapter 4 of the Users Guide for details.
    Optimal performance is a reward of about 90 per time step.
    """

    def __init__(self):
        super(World, self).__init__()
        
        self.REPORTING_PERIOD = 10 ** 4
        self.LIFESPAN = 10 ** 4
        self.REWARD_MAGNITUDE = 100.
        self.ENERGY_COST = 0.05 * self.REWARD_MAGNITUDE
        self.JUMP_FRACTION = 0.01
        self.display_state = False
        self.name = 'two dimensional grid world'
        self.announce()

        self.num_sensors = 0
        self.num_actions = 9            
        self.world_size = 5
        self.num_primitives = self.world_size ** 2
        self.MAX_NUM_FEATURES = self.num_primitives + self.num_actions
        
        self.world_state = np.array([1, 1])
        self.simple_state = self.world_state.copy()
        self.target = (3,3)
        self.obstacle = (1,1)
        self.sensors = np.zeros(self.num_sensors)

    
    def step(self, action): 
        self.timestep += 1
        action = np.round(action)
        action = action.ravel()

        self.world_state += (action[0:2] + 2 * action[2:4] - \
                             action[4:6] - 2 * action[6:8]).transpose()
        energy = np.sum(action[0:2]) + np.sum(2 * action[2:4]) + \
                 np.sum(action[4:6]) + np.sum(2 * action[6:8])
        
        """ At random intervals, jump to a random position in the world """
        if np.random.random_sample() < self.JUMP_FRACTION:
            self.world_state = np.random.random_integers(0, self.world_size, self.world_state.shape)

        """ Enforce lower and upper limits on the grid world by looping them around """
        indices = (self.world_state >= self.world_size - 0.5).nonzero()
        self.world_state[indices] -= self.world_size
        indices = (self.world_state <= -0.5).nonzero()
        self.world_state[indices] += self.world_size
        self.simple_state = np.round(self.world_state)
        primitives = np.zeros(self.num_primitives)
        primitives[self.simple_state[1] + \
                   self.simple_state[0] * self.world_size] = 1

        reward = 0
        if tuple(self.simple_state.flatten()) == self.obstacle:
            reward = - self.REWARD_MAGNITUDE
        elif tuple(self.simple_state.flatten()) == self.target:
            reward = self.REWARD_MAGNITUDE
        reward -= self.ENERGY_COST * energy
        
        self.display()
        return self.sensors, primitives, reward
    
    
    def set_agent_parameters(self, agent):
        """ Prevent the agent from forming any groups """
        agent.perceiver.NEW_GROUP_THRESHOLD = 1.0
        agent.actor.reward_min = -100.
        agent.actor.reward_max = 100.
        return


    def display(self):
        if (self.display_state):
            print self.simple_state
            
        if (self.timestep % self.REPORTING_PERIOD) == 0:
            print("world age is %s timesteps " % self.timestep)

        