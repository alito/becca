'''
Created on Jan 11, 2012

@author: brandon_rohrer
'''

import numpy as np

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
        self.num_sensors = num_sensors
        self.num_primitives = num_primitives
        self.num_actions = num_actions

        self.actions = np.zeros([self.num_actions, 1])
        
        self.timestep = 0
        
    
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
