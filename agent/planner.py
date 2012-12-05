
import utils
import viz_utils

import numpy as np

class Planner(object):

    def __init__(self, num_primitives, num_actions):
        
        """ The approximate fraction of time steps on which the 
        planner makes a random, exploratory plan.
        """
        self.EXPLORATION_FRACTION = 0.2     # real, 0 < x < 1
        
        """ The approximate fraction of time steps on which the 
        planner intentionally does nothing so that the model can observe.
        Because BECCA attends to every deliberate action and exploration,
        OBSERVATION_FRACTION should be nonzero. If it's zero, BECCA will 
        always attend its own actions, but never pay attention to their
        results.
        """
        #self.OBSERVATION_FRACTION = 0.3     # real, 0 < x < 1
        
        self.OBSERVE = True
        #debug
        self.OBSERVE_STEPS = 3
        self.observe_steps_left = self.OBSERVE_STEPS 

        """ Add just a bit of noise to the vote.
        Serves to randomize selection among nearly equal votes.
        """
        self.VOTE_NOISE = 1e-6              # real, 0 < x < 1, typically small

        self.num_primitives = num_primitives
        self.num_actions = num_actions
        self.action = np.zeros((num_actions,1))
        
        self.debug = True


    def step(self, model):
        """ Choose an action at each time step """
        
        """ First, choose a reactive action """
        """ TODO: make reactive action habit based, not reward based
        also make reactive action general """
        # debug
        # reactive_action = self.select_action(self.model, 
        #                                      self.feature_activity)

        deliberately_acted = False
        
        """ Second, choose a deliberate action (or non-action) """
        """ Only act deliberately on a fraction of the time steps """
        #if np.random.random_sample() > self.OBSERVATION_FRACTION:
        if not self.OBSERVE:
            self.OBSERVE = True
            self.observe_steps_left = self.OBSERVE_STEPS
                
            """ Occasionally explore when making a deliberate action """
            if np.random.random_sample() < self.EXPLORATION_FRACTION:
                
                self.action = self.explore()
                            
                # debug
                if self.debug:
                    print '            Exploring'
                
                """ Attend to any deliberate action """
                deliberately_acted = True
                
            else:
                """ The rest of the time, deliberately choose an action.
                Choose features as goals, in addition to action.
                """
                (self.action, goal) = self.deliberate(model)

                # debug
                if self.debug:
                    print '            Deliberating'
                
                """ Pass goal to model """ 
                model.update_goal(goal)
                
                if np.count_nonzero(self.action) > 0:
                    """ Attend to any deliberate action """
                    deliberately_acted = True
        
        else:
            self.observe_steps_left -= 1
            if self.observe_steps_left <= 0:
                self.OBSERVE = False
                
            self.action = np.zeros( self.action.shape)
            
            # debug
            if self.debug:
                print '            Observing'
                        
        #print 'action', self.action
        
        return self.action, deliberately_acted
            

    def explore(self):
        """ Forms a random, exploratory plan """
        """ TODO: form and execute multi-step plans, rather than single-step
        random action.
        """        
        """ Exploratory commands are only generated at the basic 
        action feature level. Features and higher-level commands 
        are not excited. 
        """
        """ Set all elements of the action to zero """
        action = np.zeros((self.action.shape))

        """ When exploring, randomly pick one action element to be active """
        action[np.random.random_integers(0, action.size-1)] = 1
        
        return action


    def deliberate(self, model):
        """
        Deliberate, choosing goals based on the current working memory.
        Find the transition that is best suited based on:
          1) similarity
          2) count
          3) expected reward
          4) not already having a strong goal that matches the cause
        Choose the cause of the winning transition as the goal for 
        the timestep.
        """
        """ Combine the goal-based and reward-based value, for all
        transisions, bounded by one.
        """
        """ TODO: query model, so I don't have to probe 
        its members directly. 
        """
        value = model.get_values()
        
        #value_uncertainty = model.reward_uncertainty[:model.n_transitions]
        
        """ Each transition's count and its similarity to the working memory 
        also factor in to its vote.
        """
        #count_weight = model.get_count_weight()

        similarity = model.get_context_similarities(planning=True)

        """ TODO: Add recency? This is likely to be useful when 
        rewards change over time. 
        """
        #transition_vote = value.ravel()  * similarity.ravel() 
        
        #debug--have count be a factor?
        #transition_vote = value.ravel() * similarity.ravel() * \
        #                    count_weight.ravel()
        
        #debug--penalize uncertainty
        transition_vote = 3 * value.ravel()  + similarity.ravel() 
        
        #print 'value', value[:model.num_transitions].ravel() 
        #print 'similarity', similarity[:model.num_transitions].ravel() 
        #print 'transition_vote', transition_vote[:model.num_transitions].ravel() 
        
        if transition_vote.size == 0:
            action = np.zeros(self.action.shape)
            goal = None
            return action, goal

        """ Add a small amount of noise to the votes to encourage
        variation in closely-matched options.
        """
        transition_vote += np.random.random_sample(transition_vote.shape) * \
                            self.VOTE_NOISE
        
        max_transition_index = np.argmax(transition_vote)
        
        """ Update the transition used to select this action. 
        Setting wait to True allows one extra time step for the 
        action to be executed.
        """
        model.update_transition(max_transition_index, 
                            update_strength=similarity[max_transition_index], 
                            wait=True)
        
        print 'context', model.context[:18,max_transition_index].ravel()
        print 'cause', model.cause[:18,max_transition_index].ravel()
        print 'effect', model.effect[:18,max_transition_index].ravel()
        print 'reward', model.reward_value[:,max_transition_index]
                
        print 'max_transition_index', max_transition_index
        print 'value', value.shape
        print 'value', value.ravel()[max_transition_index]
        print 'similarity', similarity [max_transition_index]
        print 'transition_vote', transition_vote[max_transition_index]
        
        goal = model.get_cause(max_transition_index)
           
        """ Separate action goals from the rest of the goal """
        action = np.zeros(self.action.shape)
        if np.size((goal[self.num_primitives: self.num_primitives + \
                         self.num_actions,:] > 0).nonzero()):
            self.deliberately_acted = True

            action[goal[self.num_primitives: self.num_primitives + \
                         self.num_actions,:] > 0] = 1
            goal[self.num_primitives: self.num_primitives + \
                 self.num_actions,:] = np.zeros((self.num_actions, 1))
    
        return action, goal

    