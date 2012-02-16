'''
Created on Jan 11, 2012

@author: brandon_rohrer
'''

import pickle

import numpy as np

from .feature_map import FeatureMap

class Agent(object):
    '''A general reinforcement learning agent, modeled on human neurophysiology 
    and performance.  It takes in a time series of 
    sensory input vectors and a scalar reward and puts out a time series 
    of motor commands.  
    New features are created as necessary to adequately represent the data.
    '''

    def __init__(self, num_sensors, num_primitives, num_actions):
        '''
        Constructor
        '''

        self.logger = logging.getLogger(self.__class__.__name__)


        self.timestep = 0
        
        self.num_sensors = num_sensors
        self.num_primitives = num_primitives
        self.num_actions = num_actions

        self.actions = np.zeros([self.num_actions, 1])

        self.GOAL_DECAY_RATE = 0.05   # real, 0 < x <= 1
        self.STEP_DISCOUNT = 0.5      # real, 0 < x <= 1

        # Rates at which the feature activity and working memory decay.
        # Setting these equal to 1 is the equivalent of making the Markovian 
        # assumption about the task--that all relevant information is captured 
        # adequately in the current state.
        self.WORKING_MEMORY_DECAY_RATE = 0.4      # real, 0 < x <= 1
        # also check out self.grouper.INPUT_DECAY_RATE, set in grouper_initialize


        self.feature_map = FeatureMap(num_sensors, num_primitives, num_actions)
        
        

        

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
        self.sensors = sensors.copy()
        self.primitives = primitives.copy()
        self.reward = reward

        self.timestep += 1   

        ''' Feature creator
        '''
        # Break inputs into groups and create new feature groups when warranted.
        grouped_input, group_added = self.grouper.step(self.sensors, self.primitives, 
                                                       self.actions, self.feature_activity)
        if group_added:
            self.add_group(grouped_input[-1].length())
            
        # Interpret inputs as features and updates feature map when appropriate.
        # Assign agent.feature_activity.
        self.update_feature_map(grouped_input)
                
        ''' Reinforcement learner
        '''
        self.previous_attended_feature = self.attended_feature
        # Attend to a single feature. Update self.attended_feature.
        self.attend()

        # Perform leaky integration on attended feature to get 
        # working memory.
        self.pre_previous_working_memory = self.previous_working_memory
        self.previous_working_memory = self.working_memory
        self.working_memory = self.integrate_state(
                    self.previous_working_memory, self.attended_feature, 
                    self.WORKING_MEMORY_DECAY_RATE)

        # Associate the reward with each transition.
        self.model.train(self.feature_activity, 
                         self.pre_previous_working_memory, 
                         self.previous_attended_feature,
                         self.reward)

        # Reactively choose an action.
        # TODO: Make reactive actions habit based, not reward based.
        #reactive_action = self.planner.select_action( ...
        #         self.model, self.feature_activity);
        
        # Only act deliberately on a fraction of the time steps.
        if np.random() > self.planner.OBSERVATION_FRACTION:
            # Occasionally explore when making a deliberate action.
            # Set self.planner.action
            if np.random() < self.planner.EXPLORATION_FRACTION:
                if __debug__:
                    print('EXPLORING')
                self.planner.explore()
            else:
                if __debug__:
                    print('DELIBERATING')

                # Deliberately choose features as goals, in addition to actions.
                self.planner.deliberate()  
        else:
            self.planner.no_action()
            
        # debug
        # self.action = util.bounded_sum(reactive_action, self.planner.action)
        self.action = self.planner.action
                
        #self.action[np.random_integers(self.num_actions)] = 1

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
        
        return()
