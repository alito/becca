
import numpy as np
import utils

class Planner(object):

    def __init__(self, num_primitives, num_actions):
        
        self.num_primitives = num_primitives
        self.num_actions = num_actions
        self.action = np.zeros((num_actions,1))
        
        """ Affects how heavily the similarity of transitions is considered during deliberation """
        self.SIMILARITY_WEIGHT = 3.         # real, 0 < x
        #self.CONFIDENCE_WEIGHT = 0.02
        
        """ Don't take any deliberate actions for (OBSERVE_STEPS - 1) time steps
        between each deliberate action.
        """
        self.OBSERVE_STEPS = 2              # integer, 0 < x, typically small
        self.observe_steps_left = self.OBSERVE_STEPS 
        self.OBSERVE = True

        """ Add just a bit of noise to the vote.
        Serves to randomize selection among nearly equal votes.
        """
        self.FITNESS_NOISE = 1e-5              # real, 0 < x < 1, typically small
        
        self.MIN_JUMP_FREQUENCY = 0.01

        """ The approximate fraction of time steps on which the 
        planner makes an exploratory plan.
        """
        self.EXPLORATION_FREQUENCY = 0.01
        self.MAX_EXPLORATION_PERIOD = 20
        self.explore_steps_left = 0
        self.EXPLORE = False
        self.exploration_goal = None
        
        self.debug = False


    def step(self, model):
        """ Choose an action at each time step """
        
        deliberately_acted = False
        
        """ Only act deliberately occasionally """
        if not self.OBSERVE:
            self.OBSERVE = True
            self.observe_steps_left = self.OBSERVE_STEPS - 1
            
            (self.action, goal) = self.deliberate(model, self.exploration_goal)  
                          
            """ Attend to any deliberate action """
            if np.count_nonzero(self.action) > 0:
                deliberately_acted = True

            if self.EXPLORE:
                """ Temporarily pursue a goal other than reward """
                self.explore_steps_left -= 1

                """ If done exploring, prepare for greedy actions the next time around """
                if self.explore_steps_left == 0:
                    self.EXPLORE = False
                    self.exploration_goal = None

                if self.debug:
                    print '            Exploring'         
                           
            else:
                """ The rest of the time, deliberately choose an action.
                Choose features as goals, in addition to action.
                """
                """ Pass goal to model """ 
                model.update_goal(goal)
                                
                """ Begin exploring the next time around? """    
                explore_val = self.EXPLORATION_FREQUENCY / (np.random.random_sample() + utils.EPSILON)
                
                if explore_val > 1:
                    self.EXPLORE = True
                    self.explore_steps_left = np.minimum(np.ceil(explore_val), self.MAX_EXPLORATION_PERIOD)
                    
                    """ Choose a new goal. The goal is the effect of a randomly selected transition. """
                    candidate_transitions = np.where(np.sum(model.effect, axis=0) > 0)
                    self.exploration_goal = model.effect[:,candidate_transitions[0][ 
                                             np.random.randint(candidate_transitions[0].size)]].copy()
                    self.exploration_goal = self.exploration_goal[:,np.newaxis]
                    
        else:
            """ if only observing """
            self.action = np.zeros( self.action.shape)
            
            if self.debug:
                print '            Observing'
                        
            self.observe_steps_left -= 1
            if self.observe_steps_left <= 0:
                self.OBSERVE = False
                
        return self.action, deliberately_acted
            

    def deliberate(self, model, exploration_goal=None):
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
        '''if model.num_transitions == 0:
            action = np.zeros(self.action.shape)
            goal = None
            return action, goal
        '''
        """ Combine the goal-based and reward-based value, for all transitions """
        if exploration_goal is None:
            #value = model.reward_value[:, :model.num_transitions]
            #uncertainty = model.reward_uncertainty[:, :model.num_transitions]
            value = model.reward_value
            uncertainty = model.reward_uncertainty
        else:
            #value = np.sum(model.effect[:, :model.num_transitions] * exploration_goal, axis=0) \
            #      / np.sum(exploration_goal, axis=0)
            #uncertainty = np.sum(model.effect_uncertainty[:, :model.num_transitions] * 
            #                        exploration_goal, axis=0) / np.sum(exploration_goal, axis=0)
            value = np.sum(model.effect * exploration_goal, axis=0) \
                  / np.sum(exploration_goal, axis=0)
            uncertainty = np.sum(model.effect_uncertainty * 
                                    exploration_goal, axis=0) / np.sum(exploration_goal, axis=0)
            uncertainty = uncertainty[np.newaxis, :]
                 
        """ Each transition's count and its similarity to the working memory 
        also factor in to its vote.
        """
        #count_weight = model.get_count_weight()

        similarity = model.get_context_similarities(planning=True)

        """ TODO: Add recency? This is likely to be useful when rewards change over time """
        transition_vote = value.ravel()  + self.SIMILARITY_WEIGHT * similarity.ravel() 
        
        """ TODO: Add confidence weighting? All else being equal, higher confidence
        transitions should be preferred, right? Initial testing shows that it doesn't help yet.
        """
        #transition_vote *= 1 - uncertainty.ravel() * self.CONFIDENCE_WEIGHT
        
        """ Add a small amount of noise to the votes to encourage variation in closely-matched options """
        noise = 1 + self.FITNESS_NOISE / np.random.random_sample(transition_vote.shape)
        transition_vote *= noise

        max_transition_index = np.argmax(transition_vote)
        
        """ Introduce periodic jumps """   
        uncertainty[:,max_transition_index]     
        if np.random.random_sample() < uncertainty[:,max_transition_index] + self.MIN_JUMP_FREQUENCY:
            action = np.zeros(self.action.shape)
            action[np.random.randint(action.size),:] = 1
            self.deliberately_acted = True
            goal = self.exploration_goal
            
            if self.debug:
                print '                    Jumping'         
                       
        else:
            """ Update the transition used to select this action. 
            Setting wait to True allows one extra time step for the action to be executed.
            """
            model.update_transition(max_transition_index, 
                                    update_strength=similarity[max_transition_index], wait=True)
            
            goal = model.get_cause(max_transition_index)
           
            """ Separate action goals from the rest of the goal """
            action = np.zeros(self.action.shape)

            if np.size((goal[self.num_primitives: self.num_primitives + self.num_actions,:] > 0).nonzero()):
                self.deliberately_acted = True
    
                action[goal[self.num_primitives: self.num_primitives + self.num_actions,:] > 0] = 1
                goal[self.num_primitives: self.num_primitives + \
                     self.num_actions,:] = np.zeros((self.num_actions, 1))
    
        return action, goal

    