
import utils
import numpy as np

class Model(object):
    """ The agent's model of transitions between states.
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
    """

    def __init__(self, num_primitives, num_actions, max_num_features, 
                 graphs=True):

        """ The threshold above which two states are similar enough to be considered a match """
        self.SIMILARITY_THRESHOLD = 0.9       # real, 0 < x < 1
        
        """ The maximum number of entries allowed in the model.
        This number is driven by the practical limitations of available
        memory and (more often) computation speed. 
        """
        self.MAX_TRANSITIONS = 10 ** 4            # integer, somewhat large
        
        """ The maximum number of features that will ever be allowed to be created """
        self.max_num_features = max_num_features  # integer, somewhat large
        
        """ How often the model is cleaned out, removing transisions that are rarely observed """
        self.CLEANING_PERIOD = 10 ** 5        # integer, somewhat large
        
        """ Lower bound on the rate at which transitions are updated """
        self.TRANSITION_UPDATE_RATE = 10 ** -1                # real, 0 < x < 1
        
        """ The number of transitions to be included in the trace context.
        The trace is used to assign credit for transitions with deferred
        effects and rewards.
        """  
        self.TRACE_LENGTH = 1                 # integer, small
        
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

        """ The total number of transitions in the model """
        self.num_transitions = 0
        
        """ The total number of features in the model """
        self.num_features = 0
        
        """ Counter tracking when to clean the model """
        self.clean_count = 0

        """ Initialize the context, cause, effect, effect_uncertainty, count, reward_value,
        and goal_value. Initialize a full model of zeros, so that all the memory is
        allocated on startup and the computer doesn't have to mess
        around with it during operation. 
        """ 
        model_shape = (self.max_num_features, 2*self.MAX_TRANSITIONS)
        self.context = np.zeros(model_shape)
        self.cause = np.zeros(model_shape)
        self.effect = np.zeros(model_shape)

        self.effect_uncertainty = np.ones(model_shape) * self.INITIAL_UNCERTAINTY
        
        thin_shape = (1, 2*self.MAX_TRANSITIONS)
        self.count = np.zeros(thin_shape)
        self.reward_value = np.zeros(thin_shape)
        self.reward_uncertainty = np.ones(thin_shape) * self.INITIAL_UNCERTAINTY
        self.goal_value = np.zeros(thin_shape)

        """ Maintain a history of the attended features and feature activity"""
        state_shape = (self.max_num_features,1)
        self.attended_feature_history = [np.zeros(state_shape)] * \
                                        self.TRACE_LENGTH
        self.feature_activity_history = [np.zeros(state_shape)] * \
                                        self.TRACE_LENGTH
        self.next_context = np.zeros(state_shape)
        self.reward_history = [0] * self.TRACE_LENGTH
        
        """ Hold on to transitions to be added or updated until their 
        future effects and rewards can be determined.
        """
        self.new_transition_q = []
        self.transition_update_q = []
        self.feature_reward_q = []

        """ These help maintain an estimate of each sensor's distribution """
        self.reward_min = utils.BIG
        self.reward_max = -utils.BIG
        self.REWARD_RANGE_DECAY_RATE = 10 ** -4
        

    def step(self, attended_feature, feature_activity, raw_reward):
        
        """ Modify the reward so that it automatically falls between
        -1 and 1, regardless of the actual reward magnitudes.
        """
        self.reward_min = np.minimum(raw_reward, self.reward_min)
        self.reward_max = np.maximum(raw_reward, self.reward_max)
        spread = self.reward_max - self.reward_min
        reward = (raw_reward - self.reward_min) / (spread + utils.EPSILON)
        
        """ account for summation during collapse """
        #reward /= 2
        
        self.reward_min += spread * self.REWARD_RANGE_DECAY_RATE
        self.reward_max -= spread * self.REWARD_RANGE_DECAY_RATE
        
        """ Update histories of attended features, feature activities, 
        and rewards. 
        """
        self.attended_feature_history.append(np.copy(attended_feature))
        self.attended_feature_history.pop(0)
        self.feature_activity_history.append(np.copy(feature_activity))
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
        self.current_reward = self.unbounded_collapse(self.reward_history)
        
        """ A collapsed collection of recent features for use in associating reward """
        self.feature_context = self.collapse(self.feature_activity_history[::-1])

        self.process_new_transitions()
        self.process_transition_updates()
        
        """ Find transitions in the library that are similar to 
        the current situation. 
        """                
        transition_match_indices, context_similarity = self.find_transition_matches()

        if len(transition_match_indices) == 0:             
            self.add_new_transition()

        else: 
            matching_transition_similarities = np.zeros(context_similarity.shape)
            matching_transition_similarities[transition_match_indices] = \
                          context_similarity[transition_match_indices]
            self.update_transition(np.argmax(matching_transition_similarities))           

        self.clean_library()        
        return
        

    def find_transition_matches(self):
        """ Check to see whether the new entry is already in the model """ 
        """ TODO: make the similarity threshold a function of the count? 
        This would allow often-observed transitions to require a closer fit, and populate
        the model space more densely in areas where it accumulates more observations.
        """
        context_similarity = self.get_context_similarities()

        """ Find which causes match. If the cause doesn't match, the transition doesn't match. """
        cause_mask = np.max(self.current_cause * self.cause[:, :self.num_transitions], axis=0)
        
        transition_similarity = context_similarity * cause_mask
        transition_match_indices = (transition_similarity >self.SIMILARITY_THRESHOLD).ravel().nonzero()[0]                                         
        return transition_match_indices, context_similarity
    
    
    def add_new_transition(self):
        
        is_new = True
        
        for i in range((len(self.new_transition_q))):
            q_context = self.new_transition_q[i][1]
            q_cause = self.new_transition_q[i][2]
            
            context_similarity = utils.similarity(self.current_context, q_context)
            cause_mask = np.max(self.current_cause * q_cause, axis=0)
            transition_similarity = context_similarity * cause_mask
            
            if transition_similarity > self.SIMILARITY_THRESHOLD:
                is_new = False
                
        if is_new:
            """ If there is no match, the just-experienced transition is
            novel. Add as a new transision in the model.
            """
            new_context = np.copy(self.current_context)
            new_cause = np.copy(self.current_cause)
            
            """ Add a new entry in the new transition queue.
            Each entry is formatted as a tuple:
            0) A timer that counts down while the effect and reward are being observed.
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
                new_effect = np.copy(self.current_effect)
                new_reward = self.current_reward
                matching_transition_index = self.num_transitions 
                
                self.context[:new_context.size, matching_transition_index] = new_context.ravel()
                self.cause[:new_cause.size, matching_transition_index] = new_cause.ravel()
                self.effect[:new_effect.size, matching_transition_index] = new_effect.ravel()
                
                self.reward_value[0, matching_transition_index] = new_reward
                self.count[0, matching_transition_index] =  1.
                self.num_transitions += 1  

        """ Remove the transitions from the queue that were added.
        This was sliced a little fancy in order to ensure that the highest
        indexed transitions were removed first, so that as the iteration
        continued the lower indices would still be accuarate.
        """
        for i in graduates[::-1]:
            self.new_transition_q.pop(i)
                
        return       


    def update_transition(self, matching_transition_index, update_strength=1.0, wait=False):
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
       
        self.transition_update_q.append([timer, matching_transition_index, update_strength])
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
                new_effect = np.copy(self.current_effect)
                new_reward = self.current_reward

                reward_difference = np.abs(new_reward - self.reward_value[0, matching_transition_index])

                """ Calculate the absolute difference of the 
                transition effect and the observation effect.
                """
                effect_difference = np.abs(new_effect[:,0]- 
                    self.effect[:new_effect.size, matching_transition_index])
                            
                self.count[0, matching_transition_index] += update_strength
                
                """ Modify the effect.
                Making the update rate a function of count allows updates to occur 
                more rapidly when there is little past experience 
                to contradict them. This facilitates one-shot learning.
                """
                update_rate_raw = (1 - self.TRANSITION_UPDATE_RATE) / \
                    self.count[0, matching_transition_index] + self.TRANSITION_UPDATE_RATE
                update_rate = min(1.0, update_rate_raw) * update_strength
                max_step_size = new_reward - self.reward_value[0, matching_transition_index]

                self.reward_value[0, matching_transition_index] += max_step_size * update_rate
                self.reward_uncertainty[0, matching_transition_index] = \
                        self.reward_uncertainty[0, matching_transition_index] *\
                        (1. - update_rate) + reward_difference * update_rate
                    
                self.effect[:new_effect.size, matching_transition_index] = \
                        self.effect[:new_effect.size, matching_transition_index] * \
                        (1. - update_rate) + new_effect[:,0] * update_rate
                        
                self.effect_uncertainty[:new_effect.size, matching_transition_index] = \
                        self.effect_uncertainty[ :new_effect.size, matching_transition_index] * \
                        (1. - update_rate) + effect_difference.ravel() * update_rate
                
        """ Remove the transitions from the queue that were added.
        This was sliced a little fancy in order to ensure that the highest
        indexed transitions were removed first, so that as the iteration
        continued the lower indices would still be accuarate.
        """
        for i in graduates[::-1]:
            self.transition_update_q.pop(i)

        return 

        
    def get_cause(self, transition_index):
        return np.copy(self.cause[:, transition_index, np.newaxis])
     
     
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
            
        n_features = context.size
        similarity = utils.similarity(context, self.context[:n_features,:], 
                                      self.num_transitions)
        return similarity
    
    
    def get_values(self):
        if self.num_transitions == 0:
            return np.zeros((0,0))
        
        """ Transform the reward to be on the interval (-1, 1) """
        reward = self.reward_value[:, :self.num_transitions]
        #mean_reward_magnitude = np.mean(np.abs(reward))
        #values = utils.map_inf_to_one(reward / (mean_reward_magnitude + utils.EPSILON))
        #return values
        return reward
    
    
    def get_value_deviations(self):
        if self.num_transitions == 0:
            return np.zeros((0,0))
        
        """ Transform the reward to be on the interval (-1, 1) """
        deviation = self.reward_uncertainty[:, :self.num_transitions]
        #mean_deviation_magnitude = np.mean(np.abs(deviation))
        #normalized_deviations = utils.map_inf_to_one(deviation / \
        #                         (mean_deviation_magnitude + utils.EPSILON))
        #return normalized_deviations
        return deviation
    
    
    def get_count_weight(self):
        return utils.map_inf_to_one(np.log(self.count[:,:self.num_transitions] + 1) / 3)
    
    
    def get_feature_salience(self, feature_activity):
       
        if self.num_transitions == 0:
            return np.zeros(feature_activity.shape)
        
        n_features = feature_activity.size
        context = self.context[:n_features,:self.num_transitions]        
        overlap = np.minimum(context, feature_activity)
        similarity = np.sum(overlap, axis=0)[np.newaxis,:]
        deviation = self.get_value_deviations()
        
        #confidence = np.maximum(1 - 2 * deviation, 0) ** 2
        confidence = 1 - deviation
        #weight = confidence * similarity * count
        weight = confidence * similarity
        feature_salience = np.sum(weight * context, axis=1) / (np.sum(weight, axis=1) + utils.EPSILON)
        feature_salience = feature_salience[:,np.newaxis]

        return feature_salience
    
    
    def collapse(self, list_to_collapse):
        """ Collapse a list of scalars or arrays into a single one, 
        giving later members of the list lower weights.
        """
        collapsed_value = list_to_collapse[0]
        
        for i in range(1,len(list_to_collapse)):            
            decayed_value = list_to_collapse[i] * ((1. - self.TRACE_DECAY_RATE) ** i)
            collapsed_value = utils.bounded_sum(collapsed_value, decayed_value)
            
        return collapsed_value
            

    def unbounded_collapse(self, list_to_collapse):
        """ Collapse a list of scalars or arrays into a single one, 
        giving later members of the list lower weights.
        Use a straight sum, rather than a bounded sum.
        """
        collapsed_value = list_to_collapse[0]
        
        for i in range(1,len(list_to_collapse)):  
            decayed_value = list_to_collapse[i] * ((1. - self.TRACE_DECAY_RATE) ** i)
            collapsed_value += decayed_value
            
        return collapsed_value
                

    def clean_library(self):

        self.clean_count += 1

        """ Clean out the model when appropriate """
        if self.num_transitions >= self.MAX_TRANSITIONS:
            self.clean_count = self.CLEANING_PERIOD + 1

        if self.clean_count > self.CLEANING_PERIOD:
            print("Cleaning up model")
            
            """ Empty these queues. Deleting library entries will
            corrupt the process of adding or updating transitions.
            """        
            self.new_transition_q = []
            self.transition_update_q = []

            self.count[0,:self.num_transitions] -=  \
                        1 / (self.count[0,:self.num_transitions] + utils.EPSILON)
            forget_indices = (self.count[0,:self.num_transitions] <= \
                                        utils.EPSILON).ravel().nonzero()[0]

            self.context = np.delete(self.context, forget_indices, 1)
            self.cause = np.delete(self.cause, forget_indices, 1)
            self.effect = np.delete(self.effect, forget_indices, 1)
            self.effect_uncertainty = np.delete(self.effect_uncertainty, forget_indices, 1)
            self.count = np.delete(self.count, forget_indices, 1)
            self.reward_value = np.delete(self.reward_value, forget_indices, 1)
            self.goal_value = np.delete(self.goal_value, forget_indices, 1)
            self.reward_uncertainty = np.delete(self.reward_uncertainty, forget_indices, 1)

            self.clean_count = 0
            self.num_transitions -= len(forget_indices)
            if self.num_transitions < 0:
                self.num_transitions = 0

            print 'Library cleaning out ', len(forget_indices), \
                    ' entries to ', self.num_transitions, ' entries '

            self.pad_model()
        return


    def pad_model(self):
        """ Pad the model (re-allocate memory space) if it has shrunk too far """
        if self.effect.shape[1] < self.MAX_TRANSITIONS:
            
            shape = (self.effect.shape[0], self.MAX_TRANSITIONS)
            thin_shape = (1, self.MAX_TRANSITIONS)
            self.context = np.hstack((self.context, np.zeros(shape)))
            self.cause  = np.hstack((self.cause, np.zeros(shape)))
            self.effect = np.hstack((self.effect, np.zeros(shape)))
            self.effect_uncertainty = np.hstack((self.effect_uncertainty, np.zeros(shape)))
            self.count = np.hstack((self.count, np.zeros(thin_shape)))
            self.reward_value = np.hstack((self.reward_value, np.zeros(thin_shape)))
            self.reward_uncertainty = np.hstack((self.reward_uncertainty, 
                                    np.ones(thin_shape) * self.INITIAL_UNCERTAINTY))
            self.goal_value = np.hstack((self.goal_value, np.zeros(thin_shape)))
        return

        
