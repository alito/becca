'''
Created on Jan 11, 2012

@author: brandon_rohrer
'''

import logging
import pickle

import numpy as np

class Agent(object):
    '''A general reinforcement learning agent, modeled on human neurophysiology 
    and performance.  It takes in a time series of 
    sensory input vectors and a scalar reward and puts out a time series 
    of motor commands.  
    New features are created as necessary to adequately represent the data.
    '''

    
    def __init__(self, num_sensors, num_primitives, num_actions, max_num_features=None):
        '''
        Constructor
        '''

        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.num_sensors = num_sensors
        self.num_primitives = num_primitives
        self.num_actions = num_actions

        self.actions = np.zeros(self.num_actions)
        
        self.timestep = 0
        

    def load(self, pickle_filename):
        loaded = False
        try:
            with open(pickle_filename, 'rb') as agent_data:
                self = pickle.load(agent_data)

            self.logger = logging.getLogger(self.__class__.__name__)
            print('Agent restored at timestep ' + str(self.timestep))

        # otherwise initializes from scratch     
        except IOError:
            # world not found
            self.logger.warn("Couldn't open %s for loading" % pickle_filename)
        except pickle.PickleError, e:
            self.logger.error("Error unpickling world: %s" % e)
        else:
            loaded = True

        return loaded

    
    def save(self, pickle_filename):
        success = False
        try:
            with open(pickle_filename, 'wb') as agent_data:
                # unset the logger before pickling and restore afterwards since it can't be pickled                
                logger = self.logger
                del self.logger
     
                pickle.dump(self, agent_data)

                self.logger = logger
            self.logger.info('agent data saved at ' + str(self.timestep) + ' time steps')

        except IOError as err:
            self.logger.error('File error: ' + str(err) + ' encountered while saving agent data')
        except pickle.PickleError as perr: 
            self.logger.error('Pickling error: ' + str(perr) + ' encountered while saving agent data')        
        else:
            success = True
            
        return success

        
    def step(self, sensors, primitives, reward):
        self.timestep += 1   

        self.actions = np.zeros(self.num_actions)
        self.actions[np.random.randint(self.num_actions)] = 1

        # debug
        """ 
        print('========================')
        print('wm:');
        self.working_memory.display()
        print('fa:');
        self.feature_activity.display()
        print('ra:');
        reactive_action.display()
        print('da:');
        self.planner.action.display()
        print('aa:');
        self.action.display()
        """
   
