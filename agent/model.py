
import state
import utils

import copy
import numpy as np

class Model(object):
    """ Contains the agent's model of transitions between states.
    The agent uses this to make predictions about what is likely
    to happen next. The ability to predict allows the agent to 
    foresee likely consequences of action it might take, helping
    it to choose good action. It can also contribute to the agent's
    attention processes by tagging feature activity patterns that 
    are unexpected.
    
    The model consists of a set of transitions. A transition
    describes an episode of the agent's experience. It mimics 
    the structure "In situation A, B happened, then C." Each 
    transision consists of four states and four scalars: 
    
    context: This state is a combination of several earlier
            attended features, each decayed according to its age
        
    cause: This state is the the attended feature that immediately
            preceded the effect. 
            
    effect: This state is the predicted result, given the context and the 
            cause. It can be a probability distribution over 
            multiple features.
            
    effect_uncertainty: This state is the expected difference
            between transition effect and the observed effect.
            
    count: This scalar tracks the number of times the transision has
            been observed, and decays slowly with time.
            It is used to represent the usefulness of
            the transition. At intervals, transitions with too low a 
            count are removed from the model.
            
    reward_value: This scalar is the expected reward, given the context
            and the cause.
            
    reward_uncertainty: This scalar is the expected difference
            between the expected reward_value and the observed reward_value.
            
    goal_value: This scalar is an internally-assigned reward value. It 
            is used during planning to prioritize intermediate 
            transisions that may have low inherent reward_value, but
            are likely to lead to rewarding transitions.
            
    Note that all attended features may be either primitives, 
    action, or higher-level features. Thus a context-cause-effect
    transition may be interpreted as state-action-state (a common
    representation in reinforcement learning methods), 
    state-state-action (a useful representation in action selection
    for planning), or state-state-state (a useful representation
    for prediction and modeling) depending on the composition
    of the cause and effect.
    
    Even though the context, cause, and effect of each transision
    are states, the operations that the model must perform are much more
    efficient if the model uses its own state-set representation, 
    rather than making an array of State objects. It allows optimized matrix
    operations, rather than for loops.
    
    """

    def __init__(self, num_primitives, num_actions, max_num_features, 
                 graphs=True):

        """ The threshold above which two states are similar enough
        to be considered a match.
        """
        self.SIMILARITY_THRESHOLD = 0.9       # real, 0 < x < 1
        
        """ The maximum number of entries allowed in the model.
        This number is driven by the practical limitations of available
        memory and (more often) computation speed. 
        """
        self.MAX_ENTRIES = 10 ** 4            # integer, somewhat large
        
        """ The maximum number of features that will ever be allowed to 
        be created.
        """
        self.MAX_NUM_FEATURES = max_num_features  # integer, somewhat large
        
        """ How often the model is cleaned out, removing transisions
        that are rarely observed. 
        """
        self.CLEANING_PERIOD = 10 ** 5        # integer, somewhat large
        
        """ Lower bound on the rate at which transitions are updated """
        self.UPDATE_RATE = 10 ** -1                # real, 0 < x < 1
        
        """ The number of transitions to be included in the trace context.
        The trace is used to assign credit for transitions with deferred
        effects and rewards.
        """  
        self.TRACE_LENGTH = 12                 # integer, small
        
        """ The factor by which the reward is decayed for each
        timestep between when it was received and the event to which
        it is assigned.
        """
        self.TRACE_DECAY_RATE = 0.6           # real, 0 < x < 1

        """ The factor by which goals are decayed for each
        timestep.
        """
        self.GOAL_DECAY_RATE = 1.0            # real, 0 < x < 1
        
        """ The initial value for uncertainty in the effect and the reward """
        self.INITIAL_UNCERTAINTY = 0.5

        """ The rate at which the reward associated with individual features
        decays toward the average value.
        """
        self.FEATURE_REWARD_DECAY_RATE = 0.01
        
        """ The total number of transitions in the model """
        self.n_transitions = 0
        
        """ The total number of features in the model """
        self.n_features = 0
        
        """ Counter tracking when to clean the model """
        self.clean_count = 0

        """ Initialize the context, cause, effect, 
        effect_uncertainty, count, reward_value,
        and goal_value.
        Initialize a full model of zeros, so that all the memory is
        allocated on startup and the computer doesn't have to mess
        around with it during operation. 
        """ 
        # TODO: create some new data structures here instead of abusing State
        self.context = state.State(num_primitives, num_actions, 
                                   max_num_features, width=2*self.MAX_ENTRIES)
        self.cause = copy.deepcopy(self.context)
        self.effect = copy.deepcopy(self.context)

        self.effect_uncertainty = self.context.ones_like()
        self.effect_uncertainty.multiply(self.INITIAL_UNCERTAINTY)
        
        thin_shape = (1, 2*self.MAX_ENTRIES)
        self.count = np.zeros(thin_shape)
        self.reward_value = np.zeros(thin_shape)
        self.reward_uncertainty = np.ones(thin_shape) * \
                                    self.INITIAL_UNCERTAINTY
        self.goal_value = np.zeros(thin_shape)

        """ Maintain a history of the attended features and feature activity"""
        self.zero_state = state.State(num_primitives, 
                                 num_actions, self.MAX_NUM_FEATURES)
        self.attended_feature_history = [copy.deepcopy(self.zero_state)] * \
                                        (self.TRACE_LENGTH)
        self.feature_activity_history = [copy.deepcopy(self.zero_state)] * \
                                        (self.TRACE_LENGTH)
        self.next_context = copy.deepcopy(self.zero_state)
        self.reward_history = [0] * (self.TRACE_LENGTH)
        self.feature_reward = copy.deepcopy(self.zero_state)
        
        """ Hold on to transitions to be added or updated until their 
        future effects and rewards can be determined """
        self.new_transition_q = []
        self.transition_update_q = []
        self.feature_reward_q = []
        

    def step(self, attended_feature, feature_activity, reward):
        
        """ Update histories of attended features, feature activities, 
        and rewards. 
        """
        self.attended_feature_history.append(copy.deepcopy(attended_feature))
        self.attended_feature_history.pop(0)
        self.feature_activity_history.append(copy.deepcopy(feature_activity))
        self.feature_activity_history.pop(0)
        self.reward_history.append(reward)
        self.reward_history.pop(0)
        
        """ Calculate the current context and cause.
        The next context is a combination of the current cause and the 
        current context.
        """ 
        self.current_context = self.next_context
        self.next_context = self.collapse(self.attended_feature_history[::-1])
        self.current_cause = self.attended_feature_history[-1]
        self.current_effect = self.collapse(self.feature_activity_history)
        #self.current_reward = self.collapse(self.reward_history)
        self.current_reward = self.unbounded_collapse(self.reward_history)
        
        """ A collapsed collection of recent features for use in 
        associating reward.
        """
        self.feature_context = self.collapse(
                                     self.feature_activity_history[::-1])
                 
        #self.associate_reward_with_features()     

        self.process_new_transitions()
        self.process_transition_updates()
        
        """ Find transitions in the library that are similar to 
        the current situation. 
        """                
        transition_match_indices, context_similarity = \
                                        self.find_transition_matches()

        if len(transition_match_indices) == 0:             
            self.add_new_transition()

        else: 
            matching_transition_similarities = np.zeros(context_similarity.shape)
            matching_transition_similarities[transition_match_indices] = \
                         context_similarity[transition_match_indices]
            self.update_transition(np.argmax(matching_transition_similarities))           

        self.clean_library()
        
        #debug
        if np.random.random_sample() < 0:
            import matplotlib.pyplot as plt
            import viz_utils
            viz_utils.visualize_model(self, 20)
            viz_utils.force_redraw()
            plt.show()

        return
        

    '''def associate_reward_with_features(self):
        """ If any feature sets are waiting to be associated, add them """
        graduates = []
        
        for i in range(len(self.feature_reward_q)):

            """ Decrement the timers in the new transition queue """
            self.feature_reward_q[i][0] = self.feature_reward_q[i][0] - 1
       
            """ Add the transitions on which the timer has counted down """
            if self.feature_reward_q[i][0] == 0:
                graduates.append(i)
                    
                features = self.feature_reward_q[i][1].features
                new_reward = self.current_reward

                # debug
                #print 'new_reward', new_reward
                #print 'features', features[:18].ravel()
                #print 'self.feature_reward.features', self.feature_reward.features[:18].ravel()

                self.feature_reward.features = \
                                 self.feature_reward.features * (1. - \
                                 self.FEATURE_REWARD_DECAY_RATE * features) + \
                                 self.FEATURE_REWARD_DECAY_RATE * features * \
                                 new_reward
                
        """ Remove the feature arrays from the queue that were processed.
        This was sliced a little fancy in order to ensure that the highest
        indexed transitions were removed first, so that as the iteration
        continued the lower indices would still be accuarate.
        """
        for i in graduates[::-1]:
            self.feature_reward_q.pop(i)
                
        """ Add the next feature array to the queue """
        timer = self.TRACE_LENGTH
        new_features = self.zero_state
        new_features.features[:self.feature_context.features.size] = \
                copy.deepcopy(self.feature_context.features)
        self.feature_reward_q.append([timer, new_features])
        
        return 
        '''
        
    def find_transition_matches(self):
        """ Check to see whether the new entry is already in the model """ 
        """ TODO: make the similarity threshold a function of the count? 
        This would
        allow often-observed transitions to require a closer fit, and populate
        the model space more densely in areas where it accumulates more
        observations.
        """
        context_similarity = self.get_context_similarities()
        
        """ Find which causes match.
        If the cause doesn't match, the transition doesn't match. 
        """
        cause_feature = self.current_cause.features.ravel().nonzero()[0]
        transition_match_indices = []
        if cause_feature is not None:
            transition_similarity = context_similarity * \
                                    self.cause.features \
                                    [cause_feature, :self.n_transitions]
                                    
            transition_match_indices = ( transition_similarity > 
                                         self.SIMILARITY_THRESHOLD). \
                                         ravel().nonzero()[0]
 
        return transition_match_indices, context_similarity
    
    
    def add_new_transition(self):
        
        is_new = True
        
        for i in range((len(self.new_transition_q))):
            q_context = self.new_transition_q[i][1]
            q_cause = self.new_transition_q[i][2]
            
            similarity = utils.similarity(self.current_context.features, 
                                q_context.features)
                                
            if (self.current_cause.equals(q_cause) and similarity > self.SIMILARITY_THRESHOLD):
                is_new = False
                
        if is_new:
            """ If there is no match, the just-experienced transition is
            novel. Add as a new transision in the model.
            """
            new_context = copy.deepcopy(self.current_context)
            new_cause = copy.deepcopy(self.current_cause)
    
            """ Add a new entry in the new transition queue.
            Each entry is formatted as a tuple:
            0) A timer that counts down while the effect and reward 
            are being observed.
            1) The current context
            2) The current cause
            """        
            timer = self.TRACE_LENGTH
            
            self.new_transition_q.append([timer, new_context, new_cause])       
        return
    
    
    def process_new_transitions(self):
        """ If any new transitions are ready to be added, add them """
        graduates = []
        
        for i in range(len(self.new_transition_q)):

            """ Decrement the timers in the new transition queue """
            self.new_transition_q[i][0] = self.new_transition_q[i][0] - 1
       
            """ Add the transitions on which the timer has counted down """
            if self.new_transition_q[i][0] == 0:
                graduates.append(i)
                    
                new_context = self.new_transition_q[i][1]
                new_cause = self.new_transition_q[i][2]
                new_effect = copy.deepcopy(self.current_effect)
                new_reward = self.current_reward
                matching_transition_index = self.n_transitions 
                
                self.context.features[:new_context.features.size, 
                                      matching_transition_index] = \
                                        new_context.features.ravel()
                self.cause.features[:new_cause.features.size, 
                                    matching_transition_index] = \
                                        new_cause.features.ravel()
                self.effect.features[:new_effect.features.size, 
                                     matching_transition_index] = \
                                        new_effect.features.ravel()
                
                self.reward_value[0, matching_transition_index] = new_reward
                self.count[0, matching_transition_index] =  1.
                self.n_transitions += 1  
                
        """ Remove the transitions from the queue that were added.
        This was sliced a little fancy in order to ensure that the highest
        indexed transitions were removed first, so that as the iteration
        continued the lower indices would still be accuarate.
        """
        for i in graduates[::-1]:
            self.new_transition_q.pop(i)
                
        return       


    def update_transition(self, matching_transition_index,
                          update_strength=1.0, wait=False):
        """ Add a new entry in the update queue.
        Each entry is formatted as a tuple:
        0) A timer that counts down while the effect and reward 
        are being observed.
        1) The index of the matching transition
        2) The update strength
        """
        """ There are two cases when a transition can be updated: when it is
        observed and when it is used by the planner to choose an action. 
        When the planner bases an action choice on a transition, an 
        extra time step is necessary in order to allow the action
        to be executed first.
        """       
        timer = self.TRACE_LENGTH
        if wait:
            timer += 1
       
        self.transition_update_q.append([timer, matching_transition_index, 
                                         update_strength])
        return
    
    
    def process_transition_updates(self):
        """ If any transitions are ready to be updated, do it """
        graduates = []
        
        """ Decrement the timers in the new transition queue """
        for i in range(len(self.transition_update_q)):
            self.transition_update_q[i][0] -= 1
            
            if self.transition_update_q[i][0] == 0:
                graduates.append(i)
                matching_transition_index = self.transition_update_q[i][1] 
                update_strength = self.transition_update_q[i][2] 
                
                """ Calculate states and values for the update """
                new_effect = copy.deepcopy(self.current_effect)
                new_reward = self.current_reward

                reward_difference = np.abs(new_reward - \
                           self.reward_value[0, matching_transition_index])

                """ Calculate the absolute difference of the 
                transition effect and the observation effect.
                """
                effect_difference = new_effect.zeros_like()
                effect_difference.set_features(np.abs(new_effect.features[:,0]- 
                       self.effect.features[:new_effect.features.size, 
                                            matching_transition_index]))
                            
                self.count[0, matching_transition_index] += update_strength
                
                """ Modify the effect.
                Making the update rate a function of count allows 
                updates to occur 
                more rapidly when there is little past experience 
                to contradict them. This facilitates one-shot learning.
                """
                update_rate_raw = (1 - self.UPDATE_RATE) / \
                    self.count[0, matching_transition_index] + self.UPDATE_RATE
                update_rate = min(1.0, update_rate_raw) * update_strength
                
                max_step_size = new_reward -  \
                                self.reward_value[0, matching_transition_index]

                self.reward_value[0, matching_transition_index] += \
                                            max_step_size * update_rate
                self.reward_uncertainty[0, matching_transition_index] = \
                        self.reward_uncertainty[0, matching_transition_index] *\
                        (1. - update_rate) + \
                        reward_difference * update_rate
                    
                self.effect.features[:new_effect.features.size, 
                                     matching_transition_index] = \
                        self.effect.features[:new_effect.features.size, 
                                             matching_transition_index] * \
                        (1. - update_rate) + new_effect.features[:,0] * \
                        update_rate
                        
                self.effect_uncertainty.features[
                    :new_effect.features.size, matching_transition_index] = \
                    self.effect_uncertainty.features[ 
                    :new_effect.features.size, matching_transition_index] * \
                    (1. - update_rate) + \
                    effect_difference.features.ravel() * update_rate
                
        """ Remove the transitions from the queue that were added.
        This was sliced a little fancy in order to ensure that the highest
        indexed transitions were removed first, so that as the iteration
        continued the lower indices would still be accuarate.
        """
        for i in graduates[::-1]:
            self.transition_update_q.pop(i)

        return 

        
    def get_cause(self, transition_index):
        transition_cause = self.next_context.zeros_like()       
        transition_cause.set_features(self.cause.features[:, transition_index])
        return transition_cause
     
     
    def update_goal(self, new_goal):
        """ Decay goals both by a fixed fraction and by the amount that 
        the feature is currently active. Experiencing a goal feature 
        causes the goal to be achieved, and the passage of time allows 
        the goal value to fade.
        """
        if new_goal != None:
            self.goal_value *= (1 - self.GOAL_DECAY_RATE)
        
        """ TODO: Add the new goal to the existing goal_value
        before decaying it. 
        """
        """ TODO: Increment the goal value of all transitions based on 
        the similarity of their effects with the goal.
        """
        return


    def get_context_similarities(self, planning=False):
        """ Return an array of similarities the same size as the 
        library, indicating the similarities between the current 
        context and the context of each transition. This format
        is useful for identifying previously seen transitions and
        for making predictions.
        
        If planning, return an array of similarities the same size as the 
        library, indicating the similarities between the *next* 
        context and the context of each transition.
        This format is useful for planning where an intermediate
        goal, including action, must be chosen.
        """
        if planning:
            context =  self.next_context
        else:
            context = self.current_context
            
        n_features = context.features.size
        similarity = utils.similarity(context.features, 
                                      self.context.features[:n_features,:], 
                                      self.n_transitions)
        return similarity
    
    
    def get_values(self):
        '''
        transition_goal_values = np.sum(self.effect.features[:self.n_features, 
                                                     :self.n_transitions] * \
                                self.goal_value[0, :self.n_transitions], axis=0)
        values = utils.bounded_sum(transition_goal_values, 
                                  self.reward_value[:, :self.n_transitions])
        
        return values[np.newaxis,:]
        '''
        
        if self.n_transitions == 0:
            return np.zeros((0,0))
        
        """ Transform the reward to be on the interval (-1, 1) """
        reward = self.reward_value[:, :self.n_transitions]
        mean_reward_magnitude = np.mean(np.abs(reward))
        values = utils.map_inf_to_one(reward / (mean_reward_magnitude + utils.EPSILON))
        
        return values
    
    
    def get_value_deviations(self):
        '''
        transition_goal_uncertainties = \
                    self.effect_uncertainty.features[:self.n_features, 
                                                     :self.n_transitions] * \
                                self.goal_value[0, :self.n_transitions]
        values = utils.bounded_sum(transition_goal_uncertainties, 
                           self.reward_uncertainty[0, :self.n_transitions])
        return values[np.newaxis,:]
        '''
             
        if self.n_transitions == 0:
            return np.zeros((0,0))
        
        """ Transform the reward to be on the interval (-1, 1) """
        deviation = self.reward_uncertainty[:, :self.n_transitions]
        mean_deviation_magnitude = np.mean(np.abs(deviation))
        normalized_deviations = utils.map_inf_to_one(deviation / \
                                (mean_deviation_magnitude + utils.EPSILON))
        
        return normalized_deviations
    
    
    def get_count_weight(self):
        return utils.map_inf_to_one(np.log(
                                    self.count[:,:self.n_transitions] + 1) / 3)

    
    def get_feature_salience(self, feature_activity):
       
        debug = False
 
        if np.random.random_sample() < 0.00:
            debug = True
            
        if self.n_transitions == 0:
            return np.zeros(feature_activity.shape)
        
        n_features = feature_activity.size
        context = self.context.features[:n_features,:self.n_transitions]
        #print 'context', context.shape
        #print 'feature_activity', feature_activity.shape

        #count = self.get_count_weight()
        #count = self.count[:,:self.n_transitions]
        #print 'count', count
        
        overlap = np.minimum(context, feature_activity)
        #print 'overlap', overlap
        
        similarity = np.sum(overlap, axis=0)[np.newaxis,:]
        #print 'similarity', similarity.ravel()
        
        deviation = self.get_value_deviations()
        #print 'deviation', deviation
        
        confidence = np.maximum(1 - 2 * deviation, 0) ** 2
        
        #weight = confidence * similarity * count
        weight = confidence * similarity
        #print 'weight', weight
        
        feature_salience = np.sum(weight * context, axis=1) / \
                        (np.sum(weight, axis=1) + utils.EPSILON)
        feature_salience = feature_salience[:,np.newaxis]
        
        #feature_salience = np.max(context * (1-deviation), axis=1)[:, np.newaxis]
        #print 'feature salience', feature_salience.shape
        if np.random.random_sample() < 0.00:
            
            for i in range(self.n_transitions):
                print 'context', self.context.features[:n_features,i].ravel() 
                print 'cause', self.cause.features[:n_features,i].ravel() 
                print 'effect', self.effect.features[:n_features,i].ravel() 
                print 'effect uncertainty', self.effect_uncertainty.features[:n_features,i].ravel() 
                print 'reward', self.reward_value[:n_features,i].ravel() 
                print 'reward uncertainty', self.reward_uncertainty[:n_features,i].ravel() 
                print 'count', self.count[:n_features,i].ravel() 
                
        if debug: 
            print 'feature_activity', feature_activity.ravel()
            print 'feature salience', feature_salience.ravel()

        return feature_salience
    
    
    def collapse(self, list_to_collapse):
        """ Collapse a list of scalars or States into a single one, 
        giving later members of the list lower weights.
        """
        if not isinstance(list_to_collapse[0], state.State):
            """ Handle the scalar list case first """
            collapsed_value = list_to_collapse[0]
            
            for i in range(1,len(list_to_collapse)):            
                decayed_value = list_to_collapse[i] * \
                                ((1. - self.TRACE_DECAY_RATE) ** i)
                collapsed_value = utils.bounded_sum(collapsed_value, 
                                                    decayed_value)
                
            return collapsed_value
            
        else:
            """ Handle the State case """
            collapsed_state = copy.deepcopy(list_to_collapse[0])
            
            for i in range(1,len(list_to_collapse)):            
                decayed_state = copy.deepcopy(list_to_collapse[i])
                decayed_state.multiply((1. - self.TRACE_DECAY_RATE) ** i)
                collapsed_state = collapsed_state.bounded_sum(decayed_state)

            return collapsed_state
    

    def unbounded_collapse(self, list_to_collapse):
        """ Collapse a list of scalars or States into a single one, 
        giving later members of the list lower weights.
        """
        if not isinstance(list_to_collapse[0], state.State):
            """ Handle the scalar list case first """
            collapsed_value = list_to_collapse[0]
            
            for i in range(1,len(list_to_collapse)):            
                decayed_value = list_to_collapse[i] * \
                                ((1. - self.TRACE_DECAY_RATE) ** i)
                collapsed_value += decayed_value
                
            return collapsed_value
            
        else:
            """ Handle the State case """
            collapsed_state = copy.deepcopy(list_to_collapse[0])
            
            for i in range(1,len(list_to_collapse)):            
                decayed_state = copy.deepcopy(list_to_collapse[i])
                decayed_state.multiply((1. - self.TRACE_DECAY_RATE) ** i)
                collapsed_state += decayed_state

            return collapsed_state
    

    def clean_library(self):

        self.clean_count += 1

        """ Clean out the model when appropriate """
        if self.n_transitions >= self.MAX_ENTRIES:
            self.clean_count = self.CLEANING_PERIOD + 1

        if self.clean_count > self.CLEANING_PERIOD:
            print("Cleaning up model")
            
            """ Empty these queues. Deleting library entries will
            corrupt the process of adding or updating transitions.
            """        
            self.new_transition_q = []
            self.transition_update_q = []

            self.count[0,:self.n_transitions] -=  \
                        1 / (self.count[0,:self.n_transitions] + utils.EPSILON)
            forget_indices = (self.count[0,:self.n_transitions] <= \
                                        utils.EPSILON).ravel().nonzero()[0]

            self.context.features = np.delete(self.context.features, 
                                                forget_indices, 1)
            self.cause.features = np.delete(self.cause.features, 
                                              forget_indices, 1)
            self.effect.features = np.delete(self.effect.features, 
                                               forget_indices, 1)
            self.effect_uncertainty.features = \
                            np.delete(self.effect_uncertainty.features, 
                            forget_indices, 1)
            self.count = np.delete(self.count, forget_indices, 1)
            self.reward_value = np.delete(self.reward_value, forget_indices, 1)
            self.goal_value = np.delete(self.goal_value, forget_indices, 1)
            self.reward_uncertainty = np.delete(self.reward_uncertainty, 
                                                forget_indices, 1)

            self.clean_count = 0
            self.n_transitions -= len(forget_indices)
            if self.n_transitions < 0:
                self.n_transitions = 0

            print 'Library cleaning out ', len(forget_indices), \
                    ' entries to ', self.n_transitions, ' entries '

            self.pad_model()
        return


    def pad_model(self):
        """ Pad the model (re-allocate memory space) 
        if it has shrunk too far. 
        """
        if self.effect.features.shape[1] < self.MAX_ENTRIES:
            
            shape = (self.effect.features.shape[0], self.MAX_ENTRIES)
            thin_shape = (1, self.MAX_ENTRIES)

            self.context.features = np.hstack((self.context.features, 
                                                 np.zeros(shape)))
            self.cause.features  = np.hstack((self.cause.features, 
                                                np.zeros(shape)))
            self.effect.features = np.hstack((self.effect.features, 
                                                np.zeros(shape)))
            self.effect_uncertainty.features = \
                                np.hstack((self.effect_uncertainty.features, 
                                np.zeros(shape)))

            self.count = np.hstack((self.count, np.zeros(thin_shape)))
            self.reward_value = np.hstack((self.reward_value, 
                                           np.zeros(thin_shape)))
            self.reward_uncertainty = np.hstack((self.reward_uncertainty, 
                                           np.ones(thin_shape) * \
                                           self.INITIAL_UNCERTAINTY))
            self.goal_value = np.hstack((self.goal_value, 
                                           np.zeros(thin_shape)))
            
        return

        
