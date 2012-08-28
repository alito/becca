
import copy
import numpy as np
import state
import utils

class Model(object):
    """ Contains the agent's model of transitions between states.
    The agent uses this to make predictions about what is likely
    to happen next. The ability to predict allows the agent to 
    foresee likely consequences of action it might take, helping
    it to choose good action. It can also contribute to the agent's
    attention processes by tagging feature activity patterns that 
    are unexpected.
    
    The model consists of a table of transitions. A transition
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

    def __init__(self, num_primitives, num_actions, graphs=True):

        """ The threshold above which two states are similar enough
        to be considered a match.
        """
        self.SIMILARITY_THRESHOLD = 0.9       # real, 0 < x < 1
        
        """ The maximum number of entries allowed in the model.
        This number is driven by the practical limitations of available
        memory and (more often) computation speed. 
        """
        self.MAX_ENTRIES = 10 ** 4            # integer, somewhat large
        
        """ How often the model is cleaned out, removing transisions
        that are rarely observed. 
        """
        self.CLEANING_PERIOD = 10 ** 5        # integer, somewhat large
        
        """ Lower bound on the rate at which transitions are updated """
        self.UPDATE_RATE = 0.1                # real, 0 < x < 1
        
        """ The number of transitions to be included in the trace context.
        The trace is used to assign credit for transitions with deferred
        effects and rewards.
        """  
        self.TRACE_LENGTH = 4                 # integer, small
        
        """ The factor by which the reward is decayed for each
        timestep between when it was received and the event to which
        it is assigned.
        """
        self.TRACE_DECAY_RATE = 0.7           # real, 0 < x < 1

        """ The factor by which goals are decayed for each
        timestep.
        """
        self.GOAL_DECAY_RATE = 1.0            # real, 0 < x < 1
        
        """ The initial value for uncertainty in the effect and the reward """
        self.INITIAL_UNCERTAINTY = 0.25

        """ The total number of transitions in the model """
        self.n_transitions = 0
        
        """ Counter tracking when to clean the model """
        self.clean_count = 0

        """ Initialize the context, cause, effect, 
        effect_uncertainty, count, reward_value,
        and goal_value.
        Initialize a full model of zeros, so that all the memory is
        allocated on startup and the computer doesn't have to mess
        around with it during operation. 
        """ 
        self.context = state.State()
        self.context.primitives = np.zeros((num_primitives, 
                                            2*self.MAX_ENTRIES))
        self.context.action = np.zeros((num_actions, 2*self.MAX_ENTRIES))
        self.context.features = []

        self.cause = copy.deepcopy(self.context)
        self.effect = copy.deepcopy(self.context)

        self.effect_uncertainty = state.State()
        self.effect_uncertainty.primitives = np.ones((num_primitives, 
                            2*self.MAX_ENTRIES)) * self.INITIAL_UNCERTAINTY
        self.effect_uncertainty.action = np.ones((num_actions, 
                            2*self.MAX_ENTRIES)) * self.INITIAL_UNCERTAINTY
        self.effect_uncertainty.features = []

        self.count = np.zeros((2*self.MAX_ENTRIES,1))
        self.reward_value = np.zeros((2*self.MAX_ENTRIES,1))
        self.reward_uncertainty = np.ones((2*self.MAX_ENTRIES,1)) * \
                                    self.INITIAL_UNCERTAINTY
        self.goal_value = np.zeros((2*self.MAX_ENTRIES,1))

        """ Maintain a history of the attended features and feature activity"""
        self.zero_state = state.State(num_sensors=0, 
                                      num_primitives=num_primitives, 
                                      num_actions=num_actions)
        self.attended_feature_history = [copy.deepcopy(self.zero_state)] * \
                                        (self.TRACE_LENGTH)
        self.feature_activity_history = [copy.deepcopy(self.zero_state)] * \
                                        (self.TRACE_LENGTH)
        self.next_context = copy.deepcopy(self.zero_state)
        self.reward_history = [0] * (self.TRACE_LENGTH)
        
        """ Hold on to transitions to be added or updated until their 
        future effects and rewards can be determined """
        self.new_transition_q = []
        self.transition_update_q = []
        

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
        self.current_reward = self.collapse(self.reward_history)
                 
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
        cause_group = None
        cause_feature = None 
               
        """ Check for matches in primitives """
        feature_match = self.current_cause.primitives.ravel().nonzero()[0]
        if feature_match.size > 0:
            cause_group = -2
            cause_feature = feature_match
               
        """ Check for matches in action """
        feature_match = self.current_cause.action.ravel().nonzero()[0]
        if feature_match.size > 0:
            cause_group = -1
            cause_feature = feature_match
               
        """ Check for matches in features """
        for group_index in range(self.n_feature_groups()):
            feature_match = self.current_cause. \
                            features[group_index].ravel().nonzero()[0]
            if feature_match.size > 0:
                cause_group = group_index
                cause_feature = feature_match
                 
        transition_match_indices = []
        if cause_group is not None:
            
            if cause_group == -2:
                transition_similarity = context_similarity * \
                                    self.cause.primitives \
                                    [cause_feature, :self.n_transitions][0]
            elif cause_group == -1:
                transition_similarity = context_similarity * \
                                    self.cause.action \
                                    [cause_feature, :self.n_transitions][0]
            else:                    
                transition_similarity = context_similarity * \
                                    self.cause.features[cause_group] \
                                    [cause_feature, :self.n_transitions][0]
                                    
            transition_match_indices = ( transition_similarity > 
                                         self.SIMILARITY_THRESHOLD). \
                                         ravel().nonzero()[0]
 
        return transition_match_indices, context_similarity
    
    
    def add_new_transition(self):
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
        
        """ Add the transitions on which the timer has counted down to 0 """
        for i in range(len(self.new_transition_q)):       
            if self.new_transition_q[i][0] == 0:
                graduates.append(i)
                    
                new_context = self.new_transition_q[i][1]
                new_cause = self.new_transition_q[i][2]
                new_effect = copy.deepcopy(self.current_effect)
                new_reward = self.current_reward
                matching_transition_index = self.n_transitions 
                self.context.primitives[:, matching_transition_index] = \
                                new_context.primitives.ravel()
                self.cause.primitives[:, matching_transition_index] = \
                                        new_cause.primitives.ravel()
                self.effect.primitives[:, matching_transition_index] = \
                                        new_effect.primitives.ravel()
                
                self.context.action[:, matching_transition_index] = \
                                        new_context.action.ravel()
                self.cause.action[:, matching_transition_index] = \
                                        new_cause.action.ravel()
                self.effect.action[:, matching_transition_index] = \
                                        new_effect.action.ravel()
                
                for group_index in range(self.n_feature_groups()):
                    self.context.features[group_index][:, matching_transition_index] = \
                                        new_context.features[group_index].ravel()
                    self.cause.features[group_index][:, matching_transition_index] = \
                                        new_cause.features[group_index].ravel()
                    self.effect.features[group_index][:, matching_transition_index] = \
                                        new_effect.features[group_index].ravel()
                                        
                self.reward_value[matching_transition_index] = new_reward
                
                self.count[matching_transition_index] =  1.
                self.n_transitions += 1  
                
            """ Decrement the timers in the new transition queue """
            self.new_transition_q[i][0] = self.new_transition_q[i][0] - 1

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
                           self.reward_value[matching_transition_index])

                """ Calculate the absolute difference of the 
                transition effect and the observation effect.
                """
                effect_difference = new_effect.zeros_like()
                effect_difference.primitives = np.abs(new_effect.primitives[:,0] - 
                       self.effect.primitives[:, matching_transition_index])
                effect_difference.action = np.abs(new_effect.action[:,0] - 
                       self.effect.action[:, matching_transition_index])
                for group_index in range(self.n_feature_groups()):
                    effect_difference.features[group_index] = \
                            np.abs(new_effect.features[group_index][:,0] - \
                            self.effect.features[group_index] \
                            [:, matching_transition_index])
                            
                self.count[matching_transition_index] += update_strength
                
                """ Modify the effect.
                Making the update rate a function of count allows 
                updates to occur 
                more rapidly when there is little past experience 
                to contradict them. This facilitates one-shot learning.
                """
                update_rate_raw = (1 - self.UPDATE_RATE) / \
                        self.count[matching_transition_index] + self.UPDATE_RATE
                update_rate = min(1.0, update_rate_raw) * update_strength
                
                max_step_size = new_reward -  \
                                self.reward_value[matching_transition_index]

                self.reward_value[matching_transition_index] += \
                                            max_step_size * update_rate
                self.reward_uncertainty[matching_transition_index] = \
                        self.reward_uncertainty[matching_transition_index] * \
                        (1. - update_rate) + \
                        reward_difference * update_rate
                    
                self.effect.primitives[:, matching_transition_index] = \
                        self.effect.primitives[:, matching_transition_index] * \
                        (1. - update_rate) + new_effect.primitives[:,0] * \
                        update_rate
                self.effect_uncertainty.primitives[:, matching_transition_index] = \
                        self.effect_uncertainty.primitives[:, matching_transition_index] * \
                        (1. - update_rate) + \
                        effect_difference.primitives * update_rate
                
                self.effect.action[:, matching_transition_index] = \
                        self.effect.action[:, matching_transition_index] * \
                        (1. - update_rate) + new_effect.action[:,0] * \
                        update_rate
                self.effect_uncertainty.action[:, matching_transition_index] = \
                        self.effect_uncertainty.action[:, matching_transition_index] * \
                        (1. - update_rate) + \
                        effect_difference.action * update_rate
                
                for group_index in range(self.n_feature_groups()):
                    self.effect.features[group_index] \
                            [:, matching_transition_index] = \
                            self.effect.features[group_index] \
                            [:, matching_transition_index] * \
                            (1. - update_rate) + \
                            new_effect.features[group_index][:,0] * update_rate
                    self.effect_uncertainty.features[group_index] \
                            [:, matching_transition_index] = \
                            self.effect_uncertainty.features[group_index] \
                            [:, matching_transition_index] * \
                            (1. - update_rate) + \
                            effect_difference.features[group_index] * \
                            update_rate
                
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
        transition_cause.primitives = self.cause.primitives[:, transition_index]
        transition_cause.action = self.cause.action[:, transition_index]
        for group_index in range(self.n_feature_groups()):
            transition_cause.features[group_index] = self.cause.features[group_index] \
                                                [:, transition_index]
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


    def get_context_similarities(self):
        """ Return an array of similarities the same size as the 
        library, indicating the similarities between the current 
        context and the context of each transition. This format
        is useful for identifying previously seen transitions and
        for making predictions.
        """
        similarity = utils.similarity(self.current_context, self.context, 
                                      self.n_transitions)
        return similarity
    
    
    def get_context_similarities_for_planning(self):
        """ Return an array of similarities the same size as the 
        library, indicating the similarities between the *next* 
        context and the context of each transition.
        This format is useful for planning where an intermediate
        goal, including action, must be chosen.
        """
        
        similarity = utils.similarity(self.next_context, self.context, 
                                      self.n_transitions)
        return similarity


    def collapse(self, state_list):
        """ Collapse a list of scalars or States into a single one, 
        giving later members of the list lower weights.
        """
        if not isinstance(state_list[0], state.State):
            """ Handle the scalar list case first """
            collapsed_value = state_list[0]
            
            for i in range(1,len(state_list)):            
                decayed_value = state_list[i] * \
                                ((1. - self.TRACE_DECAY_RATE) ** i)
                collapsed_value = utils.bounded_sum(collapsed_value, 
                                                    decayed_value)
            return collapsed_value
            
        else:
            """ Handle the State case """
            collapsed_state = copy.deepcopy(state_list[0])
            
            for i in range(1,len(state_list)):            
                decayed_state = copy.deepcopy(state_list[i]). \
                                multiply((1. - self.TRACE_DECAY_RATE) ** i)
                collapsed_state = collapsed_state.bounded_sum(decayed_state)

            return collapsed_state
    

    def clean_library(self):

        self.clean_count += 1

        eps = np.finfo(float).eps
        
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

            
            self.count[:self.n_transitions] -=  \
                                1 / self.count[:self.n_transitions]
            forget_indices = (self.count[:self.n_transitions] < eps). \
                                ravel().nonzero()[0]

            self.context.primitives = np.delete(self.context.primitives, 
                                                forget_indices, 1)
            self.cause.primitives = np.delete(self.cause.primitives, 
                                              forget_indices, 1)
            self.effect.primitives = np.delete(self.effect.primitives, 
                                               forget_indices, 1)
            self.effect_uncertainty.primitives = \
                            np.delete(self.effect_uncertainty.primitives, 
                            forget_indices, 1)

            self.context.action = np.delete(self.context.action, 
                                                forget_indices, 1)
            self.cause.action = np.delete(self.cause.action, 
                                              forget_indices, 1)
            self.effect.action = np.delete(self.effect.action, 
                                               forget_indices, 1)
            self.effect_uncertainty.action = \
                            np.delete(self.effect_uncertainty.action, 
                            forget_indices, 1)

            for group_index in range(self.n_feature_groups()):
                self.context.features[group_index] = np.delete(
                                   self.context.features[group_index], 
                                   forget_indices, 1)
                self.cause.features[group_index] = np.delete(
                                   self.cause.features[group_index], 
                                   forget_indices, 1)
                self.effect.features[group_index] = np.delete(
                                   self.effect.features[group_index], 
                                   forget_indices, 1)
                self.effect_uncertainty.features[group_index] = np.delete(
                                self.effect_uncertainty.features[group_index], 
                                forget_indices, 1)

            self.count = np.delete(self.count, forget_indices)
            self.reward_value = np.delete(self.reward_value, forget_indices)
            self.goal_value = np.delete(self.goal_value, forget_indices)
            self.reward_uncertainty = np.delete(self.reward_uncertainty, 
                                                forget_indices)

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
        if self.effect.primitives.shape[1] < self.MAX_ENTRIES * 1.1:
            
            size = (self.effect.primitives.shape[0], self.MAX_ENTRIES)
            self.context.primitives = np.hstack((self.context.primitives, 
                                                 np.zeros(size)))
            self.cause.primitives  = np.hstack((self.cause.primitives, 
                                                np.zeros(size)))
            self.effect.primitives = np.hstack((self.effect.primitives, 
                                                np.zeros(size)))
            self.effect_uncertainty.primitives = \
                                np.hstack((self.effect_uncertainty.primitives, 
                                np.zeros(size)))

            size = (self.effect.action.shape[0], self.MAX_ENTRIES)
            self.context.action = np.hstack((self.context.action, 
                                              np.zeros(size)))
            self.cause.action  = np.hstack((self.cause.action, 
                                             np.zeros(size)))
            self.effect.action = np.hstack((self.effect.action, 
                                             np.zeros(size)))
            self.effect_uncertainty.action = \
                                np.hstack((self.effect_uncertainty.action, 
                                np.zeros(size)))

            for group_index in range(self.n_feature_groups()):
                size = (self.effect.features[group_index].shape[0], 
                        self.MAX_ENTRIES)
                self.context.features[group_index] = np.hstack((
                        self.context.features[group_index], np.zeros(size)))
                self.cause.features[group_index]  = np.hstack((
                        self.cause.features[group_index], np.zeros(size)))
                self.effect.features[group_index] = np.hstack((
                        self.effect.features[group_index], np.zeros(size)))
                self.effect_uncertainty.features[group_index] = np.hstack((
                        self.effect_uncertainty.features[group_index], 
                        np.zeros(size)))

  
            self.count = np.hstack((self.count, np.zeros(self.MAX_ENTRIES)))
            self.reward_value = np.hstack((self.reward_value, 
                                           np.zeros(self.MAX_ENTRIES)))
            self.reward_uncertainty = np.hstack((self.reward_uncertainty, 
                                           np.ones(self.MAX_ENTRIES) * \
                                           self.INITIAL_UNCERTAINTY))
            self.goal_value = np.hstack((self.goal_value, 
                                           np.zeros(self.MAX_ENTRIES)))

        return
    

    def add_group(self, n_features):
        size = (n_features, self.cause.action.shape[1])
        self.context.features.append(np.zeros(size))
        self.cause.features.append(np.zeros(size))
        self.effect.features.append(np.zeros(size))
        self.effect_uncertainty.features.append(np.ones(size) \
                                        * self.INITIAL_UNCERTAINTY)
                                        
        self.next_context.add_group(n_features)
        
        for group_index in range(len(self.attended_feature_history)):
            self.attended_feature_history[group_index].\
                                                add_group(n_features)
            self.feature_activity_history[group_index].\
                                                add_group(n_features)
            
        for group_index in range(len(self.new_transition_q)):
            self.new_transition_q[group_index][1].add_group(n_features)
            self.new_transition_q[group_index][2].add_group(n_features)
        
                    
    def n_feature_groups(self):
        return len(self.context.features)
