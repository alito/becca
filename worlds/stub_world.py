'''
Created on Jan 11, 2012

@author: brandon_rohrer

The only methods that a world is required to implement are:

    step() -advances the world by one time step
    
    final_performance() -returns a performance value for the 
        agent in the world once the termination condition for the 
        world has been met

A number of methods for internal use have been found to be convenient as well:

    set_agent_parameters() -configures BECCA
        parameters specifically for the task at hand, when necessary

    display() -displays progress to the user

    log() -logs task progress and saves the current state of the world and 
        agent to disk
'''

import becca.agent
import pickle
import numpy as np
import matplotlib.pyplot as plt

class StubWorld(object):
    '''
    the base class for creating a new world
    '''

    def __init__(self):
        ''' default constructor
        '''
    def initialize(self):
        ''' performs initialization, but gets around the fact that __init__()
        can't return objects
        '''
        self.filename_prefix = "stub"
        self.agent_filename = self.filename_prefix + "_agent.pickle";
        self.world_filename = self.filename_prefix + "_world.pickle";

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
            self.num_sensors = 1
            self.num_primitives = 1
            self.num_actions = 1

            self.timestep = 0
            self.REPORTING_PERIOD = pow(10, 3)
            self.BACKUP_PERIOD = pow(10, 3)
            self.LIFESPAN = pow(10, 4)
            
            self.sensors = np.zeros(self.num_sensors)
            self.primitives = np.zeros(self.num_primitives)
            self.actions = np.zeros(self.num_actions)
            self.reward = 0
            
            self.world_state = 0            
            self.cumulative_reward = 0
            self.reward_history = np.array([]);
            
            self.display_features_flag = False;
            
            plt.figure(1) 
            plt.clf
            plt.xlabel('block (' + str(self.REPORTING_PERIOD) +  ' time steps per block)');
            plt.ylabel('reward per block');
            plt.ion()
            
            agent = becca.agent.Agent(self.num_sensors, self.num_primitives, self.num_actions)
        
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
            print('Display the world state: world timestep ' + str(self.timestep))
            
        if (np.mod(self.timestep, self.REPORTING_PERIOD) == 0):
            self.reward_history = np.append(self.reward_history, self.cumulative_reward)
            self.cumulative_reward = 0;
            plt.plot(self.reward_history)


    def log(self, agent):
        ''' logs the state of the world into a history that can be used to
        evaluate and understand BECCA's behavior
        '''
        self.cumulative_reward += self.reward;
        
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
        self.timestep = self.timestep + 1 
        self.action = action.copy()

        print('Write your own damn world.step() method!')

        self.log()
        self.display()
        return()
 
        
    def final_performance(self):
        '''
        when the world terminates, this returns the performance 
        of the agent, a real value between -1 and 1. Before reaching
        the termination condition, it returns a value less than -1.
        Any terminating activities or reports should be included
        in this method too.
        '''
        if (self.timestep > self.LIFESPAN):
            performance = np.mean(self.reward_history[-3:]) / self.REPORTING_PERIOD
            plt.ioff()
            plt.show()
            
            assert(performance >= -1.)
            return(performance)
        
        return(-2)
    