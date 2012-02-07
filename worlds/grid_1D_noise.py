'''
Created on Jan 11, 2012

@author: brandon_rohrer
'''
import random

import stub_world
import agent_stub as ag
import pickle
import numpy as np
#import matplotlib.pyplot as plt

class World(stub_world.StubWorld):
    ''' grid_1D_noise.World
    One-dimensional grid task with noise

    In this task, the agent steps forward and backward along three positions 
    on a line. The second position is rewarded (1/2) and the first and third
    positions are punished (-1/2). Also, any actions are penalized (-1/10).
    It also includes some basic feature inputs that are pure noise.

    Optimal performance is between 0.3 and 0.35 reward per time step.
    '''
    
    def __init__(self):
        ''' default constructor
        '''
    def initialize(self):
        ''' performs initialization, but gets around the fact that __init__()
        can't return objects
        '''
        self.filename_prefix = "grid_1D_noise"
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

            self.timestep = 0
            self.noise_inputs = 7
            self.num_sensors = 1
            self.num_real_features = 3
            self.num_actions = 3
            self.num_primitives = self.noise_inputs + self.num_real_features
            
            self.REPORTING_PERIOD = pow(10, 3)
            self.BACKUP_PERIOD = pow(10, 3)
            self.LIFESPAN = pow(10, 4)
            self.ENERGY_PENALTY = 0.1
            
            self.sensors = np.zeros(self.num_sensors)
            self.primitives = np.zeros(self.num_primitives)
            self.actions = np.zeros(self.num_actions)
            self.reward = 0
            
            self.world_state = 0            
            self.cumulative_reward = 0
            self.reward_history = np.array([])
            
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

        if (self.timestep % self.BACKUP_PERIOD) == 0:
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
        energy = np.sum(action)

        self.world_state += action[0] - action[1]
        self.world_state = np.round(self.world_state)
        self.world_state = min(self.world_state, 2)
        self.world_state = max(self.world_state, 0)

        real_features = np.zeros((self.num_real_features,))
        real_features[int(self.world_state)] = 1

        noise = np.round(np.random.random_sample((self.num_real_features,)))
        self.basic_feature_input = np.hstack((real_features, noise))

        self.reward = -0.5
        if int(self.world_state) == 2:
            self.reward = 0.5

        self.reward -= energy * self.ENERGY_PENALTY
        
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
            
            return performance
        
        return -2
