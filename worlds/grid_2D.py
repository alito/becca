import numpy as np
from .base_world import World as BaseWorld

class World(BaseWorld):
    """ grid_2D.World
    Two-dimensional grid task

    In this task, the agent steps North, South, East, or West in a
    5 x 5 grid-world. Position (4,4) is rewarded (1/2) and (2,2) is
    punished (-1/2). There is also a penalty of -1/20 for each horizontal
    or vertical step taken.
    Horizonal and vertical positions are reported
    separately as basic features, rather than raw sensory inputs.

    This is intended to be a
    simple-as-possible-but-slightly-more-interesting-
    that-the-one-dimensional-task task for troubleshooting BECCA.

    Optimal performance is between 0.3 and 0.35 reward per time step.
    """

    def __init__(self):
                
        super(World, self).__init__()
        
        self.REPORTING_PERIOD = 10 ** 3
        self.LIFESPAN = 5 * 10 ** 4
        self.REWARD_MAGNITUDE = 0.5
        self.ENERGY_COST = 0.05
        self.display_state = False

        self.num_sensors = 1
        self.num_actions = 9            
        self.world_size = 5
        self.num_primitives = self.world_size ** 2
        self.world_state = np.array([1, 1])
        self.simple_state = self.world_state.copy()

        self.target = (3,3)
        self.obstacle = (1,1)

        self.sensors = np.zeros(self.num_sensors)

    
    def step(self, action): 
        """ Advance the World by one timestep """

        self.timestep += 1
        
        action = np.round(action)

        self.world_state += (action[0:2] + 2 * action[2:4] - \
                             action[4:6] - 2 * action[6:8]).transpose()

        energy = np.sum(action[0:2]) + np.sum(2 * action[2:4]) + \
                 np.sum(action[4:6]) - np.sum(2 * action[6:8])
        

        """ Enforce lower and upper limits on the grid world by 
        looping them around. It actually has a toroidal topology.
        """
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
        agent.grouper.NEW_GROUP_THRESHOLD = 1.0
        

    def display(self):
        """ Provide an intuitive display of the current state of the World 
        to the user.
        """
        if (self.display_state):
            print self.simple_state
            
        if (self.timestep % self.REPORTING_PERIOD) == 0:
            print("world age is %s timesteps " % self.timestep)

        