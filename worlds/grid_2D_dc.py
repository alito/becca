'''
Created on Jan 11, 2012

@author: brandon_rohrer
'''
import stub_world
import agent_stub as ag
import pickle
import numpy as np
#import matplotlib.pyplot as plt

class World(stub_world.StubWorld):
    ''' grid_2D_dc.World
    Two-dimensional grid task

    dc stands for decoupled. It's just like the task_grid_2D task except that 
    the two dimensions are decoupled. The basic feature vector represents a
    row and a column separately, not coupled together.

    In this task, the agent steps North, South, East, or West in a
    5 x 5 grid-world. Position (4,4) is rewarded (1/2) and (2,2) is
    punished (-1/2).  There is also a penalty of -1/20 for each horizontal
    or vertical step taken. Horizonal and vertical positions are reported
    separately as basic features, rather than raw sensory inputs.

    This is intended to be a
    simple-as-possible-but-slightly-more-interesting-
    that-the-one-dimensional-task task for troubleshooting BECCA.

    Optimal performance is between 0.3 and 0.35.

    '''

    def __init__(self):
        ''' default constructor
        '''
    def initialize(self):
        ''' performs initialization, but gets around the fact that __init__()
        can't return objects
        '''
        self.filename_prefix = "grid_2D_dc"
        self.agent_filename = self.filename_prefix + "_agent.pickle"
        self.world_filename = self.filename_prefix + "_world.pickle"

        # if there is a stored version of the world and agent, loads it
        try:
            with open(self.world_filename, 'rb') as world_data:
                self = pickle.load(world_data)       
            with open(self.agent_filename, 'rb') as agent_data:
                agent = pickle.load(agent_data)  
            print('World restored at timestep ' + str(self.timestep))
                
        # otherwise initializes from scratch     
        except:     
            print('Initializing world and agent...')


            self.REPORTING_PERIOD = 10 ** 3
            self.BACKUP_PERIOD = 10 ** 3
            self.LIFESPAN = 10 ** 4
            self.ENERGY_PENALTY = 0.05

            self.timestep = 0
            self.num_sensors = 1
            self.num_actions = 9            
            self.world_size = 5
            self.num_primitives = self.world_size * 2
            self.world_state = np.array([1, 1])
            self.simple_state = self.world_state.copy()

            self.target = (3,3)
            self.obstacle = (1,1)
            
            self.sensors = np.zeros(self.num_sensors)
            self.primitives = np.zeros(self.num_primitives)
            self.actions = np.zeros(self.num_actions)
            self.reward = 0
            
            self.cumulative_reward = 0
            self.reward_history = np.array([])
            self.motor_output_history = np.array([])            
            
            self.display_features_flag = False
            """
            plt.figure(1) 
            plt.clf
            plt.xlabel('block (' + str(self.REPORTING_PERIOD) +  ' time steps per block)');
            plt.ylabel('reward per block');
            plt.ion()
            """
            agent = ag.Agent(self.num_sensors, self.num_primitives, self.num_actions)
        
        self.set_agent_parameters(agent)
        return(self, agent)


    def set_agent_parameters(self, agent):
        ''' sets parameters in the BECCA agent that are specific to a particular world.
        Strictly speaking, this method violates the minimal interface between the 
        agent and the world (observations, action, and reward). Ideally, it will 
        eventually become obselete. As BECCA matures it will be able to handle 
        more tasks without changing its parameters.
        '''
        pass

    
    def display(self):
        ''' provides an intuitive display of the current state of the World 
        to the user
        '''
        if (self.display_features_flag):
            state_img = ['.'] * self.num_primitives
            state_img[self.world_state] = 'O'
            print('world timestep ' + str(self.timestep) + '  ' + ''.join(state_img))
            
        if (np.mod(self.timestep, self.REPORTING_PERIOD) == 0):
            self.reward_history = np.append(self.reward_history, self.cumulative_reward)
            self.cumulative_reward = 0
            #plt.plot(self.reward_history)


    def log(self, agent):
        ''' logs the state of the world into a history that can be used to
        evaluate and understand BECCA's behavior
        '''
        self.cumulative_reward += self.reward
        
        if (np.mod(self.timestep, self.BACKUP_PERIOD) == 0):
            # stores the world and the agent
            try:
                with open(self.world_filename, 'wb') as world_data:
                    pickle.dump(self, world_data)       
                with open(self.agent_filename, 'wb') as agent_data:
                    pickle.dump(agent, agent_data)  
                print('agent data saved at ' + str(self.timestep) + ' time steps')
                    
            except IOError as err:
                print('File error: ' + str(err) + ' encountered while saving data')
            except pickle.PickleError as perr: 
                print('Pickling error: ' + str(perr) + ' encountered while saving data')
        
        
    def step(self, action, agent): 
        ''' advances the World by one timestep.
        Accepts agent as an argument only so that it can occasionally backup
        the agent's state to disk.
        '''
        self.timestep += 1
        
        action = np.round(action)

        self.world_state += (action[0:2] + 2 * action[2:4] - action[4:6] - 2 * action[6:8]).transpose()

        energy = np.sum(action[0:2]) + np.sum(2 * action[2:4]) + np.sum(action[4:6]) - np.sum(2 * action[6:8])
        

        #enforces lower and upper limits on the grid world by looping them around.
        #It actually has a toroidal topology.
        indices = self.world_state >= self.world_size - 0.5
        self.world_state[indices] -= self.world_size

        indices = self.world_state <= -0.5
        self.world_state[indices] += self.world_size

        self.simple_state = np.round(self.world_state)

        self.basic_feature_input = np.zeros((self.num_primitives,))
        self.basic_feature_input[self.simple_state[0]] = 1
        self.basic_feature_input[self.simple_state[1] + self.world_size] = 1

        self.reward = 0
        if tuple(self.simple_state.flatten()) == self.obstacle:
            self.reward = -0.5
        elif tuple(self.simple_state.flatten()) == self.target:
            self.reward = 0.5

        self.reward -= self.ENERGY_PENALTY * energy

        
        self.log(agent)
        self.display()
        

        
    def final_performance(self):
        ''' When the world terminates, this returns the average performance 
        of the agent on the last 3 blocks.
        '''
        if (self.timestep > self.LIFESPAN):
            performance = np.mean(self.reward_history[-3:]) / self.REPORTING_PERIOD
            #plt.ioff()
            #plt.show()
            
            assert(performance >= -1.)
            return performance
        
        return -2
