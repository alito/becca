
import copy
import numpy as np
import state
import utils

class Model(object):
    """ Contains the agent's model of transitions between states.
    The agent uses this to make predictions about what is likely
    to happen next. The ability to predict allows the agent to 
    foresee likely consequences of actions it might take, helping
    it to choose good actions. It can also contribute to the agent's
    attention processes by tagging feature activity patterns that 
    are unexpected.
    
    The model consists of a table of transitions. A transition
    describes an episode of the agent's experience. It mimics 
    the structure "In situation A, B happened, then C." Each 
    transision consists of three states and three scalars: 
    a context, a cause, an effect, a count, a reward_value,
    and a goal_value. 
    
    context: This state is a combination of several earlier
            attended features, each decayed according to its age
        
    cause: This state is the the attended feature that immediately
            preceded the effect. 
            
    effect: This state is the predicted result, given the context and the 
            cause. It can be a probability distribution over 
            multiple features.
            
    count: This scalar tracks the number of times the transision has
            been observed, and decays slowly with time.
            It is used to represent the usefulness of
            the transition. At intervals, transitions with too low a 
            count are removed from the model.
            
    reward_value: This scalar is the expected reward, given the context
            and the cause.
            
    goal_value: This scalar is an internally-assigned reward value. It 
            is used during planning to prioritize intermediate 
            transisions that may have low inherent reward_value, but
            are likely to lead to rewarding transitions.
            
    Note that all attended features may be either primitives, 
    actions, or higher-level features. Thus a context-cause-effect
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
        
        """ Lower bound on the rate at which effects are updated """
        self.UPDATE_RATE = 0.1                # real, 0 < x < 1
        
        """ The number of transitions to be included in the trace context.
        The trace is used to assign credit for transitions with deferred
        effects and rewards.
        """  
        self.TRACE_LENGTH = 4               # integer, small
        
        """ The factor by which the reward is decayed for each
        timestep between when it was received and the event to which
        it is assigned.
        """
        self.TRACE_DECAY_RATE = 0.2           # real, 0 < x < 1

        """ The factor by which goals are decayed for each
        timestep.
        """
        self.GOAL_DECAY_RATE = 0.2           # real, 0 < x < 1

        """ The total number of transitions in the model """
        self.n_transitions = 0
        
        """ Counter tracking when to clean the model """
        self.clean_count = 0

        """ Initialize the context, cause, effect, count, reward_value,
        and goal_value.
        Initialize a full model of zeros, so that all the memory is
        allocated on startup and the computer doesn't have to mess
        around with it during operation. 
        """ 
        self.context = state.State()
        self.context.primitives = np.zeros((num_primitives, 
                                            2*self.MAX_ENTRIES))
        self.context.actions = np.zeros((num_actions, 2*self.MAX_ENTRIES))
        self.context.features = []

        self.cause = copy.deepcopy(self.context)
        self.effect = copy.deepcopy(self.context)

        self.count = np.zeros((2*self.MAX_ENTRIES,1))
        self.reward_value = np.zeros((2*self.MAX_ENTRIES,1))
        self.goal_value = np.zeros((2*self.MAX_ENTRIES,1))

        self.trace_index = np.ones((self.TRACE_LENGTH,1))
        self.trace_reward = np.zeros((self.TRACE_LENGTH,1))


    def step(self, new_context, new_cause, new_effect, reward):
        """ Take in new_context, new_cause and new_effect 
        to train the model.
        """
        transition_match_indices, context_similarity = \
                    self.find_transition_matches(new_context, new_cause)

        if len(transition_match_indices) == 0:             
            
            # debug
            '''
            if np.random.random_sample() < 0.02:
                import viz_utils
                import matplotlib.pyplot as plt
                fig = plt.figure('new transition')
                ax = fig.add_subplot(1,1,1)
    
                viz_utils.visualize_state(new_cause, "new_cause",
                                          y_min=1.25, y_max=1.75, axes=ax)
                viz_utils.visualize_state(new_effect, "new_effect",
                                          y_min=0.25, y_max=0.75, axes=ax)
                viz_utils.visualize_state(new_context,  "new_context",
                                          y_min=2.25, y_max=2.75, axes=ax)
                                          
                plt.plot(0, 0, color='black') 
    
                plt.show()
            '''
            
            matching_transition_index, reward_update_rate = \
                    self.add_new_transition(new_context, 
                                            new_cause, 
                                            new_effect)
        else:            
            matching_transition_index, reward_update_rate = \
                    self.update_matching_transitions(context_similarity, 
                                                     transition_match_indices,
                                                     new_effect)
                    
            # debug  
            '''    
            if np.random.random_sample() < 0.02:
                import viz_utils
                import matplotlib.pyplot as plt
                fig = plt.figure('new transition')
                ax = fig.add_subplot(1,1,1)
    
                viz_utils.visualize_state(new_cause, "new_cause",
                                          y_min=1.25, y_max=1.75, axes=ax)
                viz_utils.visualize_state(new_effect, "new_effect",
                                          y_min=0.25, y_max=0.75, axes=ax)
                viz_utils.visualize_state(new_context,  "new_context",
                                          y_min=2.25, y_max=2.75, axes=ax)
                                          
                plt.plot(0, 0, color='black') 
    
                viz_utils.visualize_transition(self, matching_transition_index)
                plt.show()        
            '''
            
        self.update_reward(reward_update_rate, matching_transition_index, 
                           reward)               
        
        self.clean_library()
        
        return
        
        
    def find_transition_matches(self, new_context, new_cause):
        """ Check to see whether the new entry is already in the model """ 
        """ TODO: make the similarity threshold a function of the count? 
        This would
        allow often-observed transitions to require a closer fit, and populate
        the model space more densely in areas where it accumulates more
        observations.
        """
        context_similarity = utils.similarity(new_context, self.context, 
                                              self.n_transitions)
        
        """ Find which causes match.
        If the cause doesn't match, the transition doesn't match. 
        """
        cause_group = None
        cause_feature = None 
               
        """ Check for matches in primitives """
        feature_match = new_cause.primitives.ravel().nonzero()[0]
        if feature_match.size > 0:
            cause_group = -2
            cause_feature = feature_match
               
        """ Check for matches in actions """
        feature_match = new_cause.actions.ravel().nonzero()[0]
        if feature_match.size > 0:
            cause_group = -1
            cause_feature = feature_match
               
        """ Check for matches in features """
        for group_index in range(self.n_feature_groups()):
            feature_match = new_cause.features[group_index].ravel().nonzero()[0]
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
                                    self.cause.actions \
                                    [cause_feature, :self.n_transitions][0]
            else:                    
                transition_similarity = context_similarity * \
                                    self.cause.features[cause_group] \
                                    [cause_feature, :self.n_transitions][0]
                                    
            transition_match_indices = ( transition_similarity > 
                                         self.SIMILARITY_THRESHOLD). \
                                         ravel().nonzero()[0]
 
        return transition_match_indices, context_similarity
    
    
    def add_new_transition(self, new_context, new_cause, new_effect):
        """ If there is no match, the just-experienced transition is
        novel. Add is a new transision in the model.
        """
        matching_transition_index = self.n_transitions

        self.context.primitives[:, self.n_transitions] = \
                                new_context.primitives.ravel()
        self.cause.primitives[:, self.n_transitions] = \
                                new_cause.primitives.ravel()
        self.effect.primitives[:, self.n_transitions] = \
                                new_effect.primitives.ravel()

        self.context.actions[:, self.n_transitions] = \
                                new_context.actions.ravel()
        self.cause.actions[:, self.n_transitions] = \
                                new_cause.actions.ravel()
        self.effect.actions[:, self.n_transitions] = \
                                new_effect.actions.ravel()
    
        for group_index in range(self.n_feature_groups()):
            self.context.features[group_index][:, self.n_transitions] = \
                                new_context.features[group_index].ravel()
            self.cause.features[group_index][:, self.n_transitions] = \
                                new_cause.features[group_index].ravel()
            self.effect.features[group_index][:, self.n_transitions] = \
                                new_effect.features[group_index].ravel()
        
        self.count[matching_transition_index] =  1.
        reward_update_rate = 1.
        self.n_transitions += 1  
        
        return matching_transition_index, reward_update_rate       

    
    def update_matching_transitions(self, context_similarity, 
                                    transition_match_indices, new_effect):
        """ Only consider matching transitions. Use transition_match_indices
        as a mask for context_similarity.
        """
        transition_similarity = np.zeros(context_similarity.shape)
        transition_similarity[transition_match_indices] = \
                    context_similarity[transition_match_indices]
                    
        """ Increment the nearest entry """
        matching_transition_index = np.argmax(transition_similarity)                    
        self.count[matching_transition_index] += 1
        
        """ Modify the effect.
        Making the update rate a function of count allows updates to occur
        more rapidly when there is little past experience 
        to contradict them. This facilitates one-shot learning.
        """
        reward_update_rate = (1 - self.UPDATE_RATE) / \
                    self.count[matching_transition_index] + self.UPDATE_RATE
        reward_update_rate = min(1.0, reward_update_rate)
        
        self.effect.primitives[:, matching_transition_index] = \
                self.effect.primitives[:, matching_transition_index] * \
                (1 - reward_update_rate) + new_effect.primitives[:,0] * \
                reward_update_rate
        
        self.effect.actions[:, matching_transition_index] = \
                self.effect.actions[:, matching_transition_index] * \
                (1 - reward_update_rate) + new_effect.actions[:,0] * \
                reward_update_rate
        
        for group_index in range(self.n_feature_groups()):
            self.effect.features[group_index] \
                    [:, matching_transition_index] = \
                    self.effect.features[group_index] \
                    [:, matching_transition_index] * \
                    (1 - reward_update_rate) + \
                    new_effect.features[group_index][:,0] * reward_update_rate
        # TODO: update context as well?
        
        return matching_transition_index, reward_update_rate

    
    def update_reward(self, reward_update_rate, matching_transition_index, 
                      reward):
        """ Perform credit assignment on the trace """
        """ Update the transition trace """
        
        self.trace_index = np.vstack((self.trace_index, 
                            matching_transition_index * np.ones((1,1))))
        self.trace_index = self.trace_index[1:,:]

        """ Update the reward trace """
        self.trace_reward = np.vstack((self.trace_reward, 
                                       reward * np.ones((1,1))))
        self.trace_reward = self.trace_reward[1:,:]

        credit = self.trace_reward[0,0]
        for trace_index in range(1, self.TRACE_LENGTH):
            credit = utils.bounded_sum( credit, 
                                self.trace_reward[trace_index,0] *  \
                                (1 - self.TRACE_DECAY_RATE) ** trace_index)

        """ Update the reward associated with the last entry in the trace """
        update_index = self.trace_index[0,0]
        max_step_size = credit - self.reward_value[update_index]

        self.reward_value[update_index] += max_step_size * reward_update_rate
     
     
    def update_goal(self, new_goal):
        """ Decay goals both by a fixed fraction and by the amount that 
        the feature is currently active. Experiencing a goal feature 
        causes the goal to be achieved, and the passage of time allows 
        the goal value to fade.
        """
        self.goal_value *= (1 - self.GOAL_DECAY_RATE)
        
        """ TODO: Increment the goal value of all transitions based on 
        the similarity of their effects with the goal.
        """

        return


    def clean_library(self):

        self.clean_count += 1

        eps = np.finfo(float).eps
        
        """ Clean out the model when appropriate """
        if self.n_transitions >= self.MAX_ENTRIES:
            self.clean_count = self.CLEANING_PERIOD + 1

        if self.clean_count > self.CLEANING_PERIOD:
            print("Cleaning up model")
            
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

            self.context.actions = np.delete(self.context.actions, 
                                                forget_indices, 1)
            self.cause.actions = np.delete(self.cause.actions, 
                                              forget_indices, 1)
            self.effect.actions = np.delete(self.effect.actions, 
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

            self.count = np.delete(self.count, forget_indices)
            self.reward_value = np.delete(self.reward_value, forget_indices)
            self.goal_value = np.delete(self.goal_value, forget_indices)

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

            size = (self.effect.actions.shape[0], self.MAX_ENTRIES)
            self.context.actions = np.hstack((self.context.actions, 
                                              np.zeros(size)))
            self.cause.actions  = np.hstack((self.cause.actions, 
                                             np.zeros(size)))
            self.effect.actions = np.hstack((self.effect.actions, 
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

  
            self.count = np.hstack((self.count, np.zeros(self.MAX_ENTRIES)))
            self.reward_value = np.hstack((self.reward_value, 
                                           np.zeros(self.MAX_ENTRIES)))
            self.goal_value = np.hstack((self.goal_value, 
                                           np.zeros(self.MAX_ENTRIES)))

        return
    

    def add_fixed_group(self, n_features):
        size = (n_features, self.cause.actions.shape[1])
        self.context.features.append(np.zeros(size))
        self.cause.features.append(np.zeros(size))
        self.effect.features.append(np.zeros(size))

        '''
        def add_group(self):
        size = (0, self.cause.actions.shape[1])
        self.context.features.append(np.zeros(size))
        self.cause.features.append(np.zeros(size))
        self.effect.features.append(np.zeros(size))
        '''

        '''
        def add_feature(self, nth_group):
        """ Add a feature to the nth group of the model """
        
        self.context.features[nth_group] = \
                    np.vstack((self.context.features[nth_group],  
                    np.zeros(self.context.features[nth_group].shape[1])))
        self.cause.features[nth_group]   = \
                    np.vstack((self.cause.features[nth_group],  
                    np.zeros(self.cause.features[nth_group].shape[1])))
        self.effect.features[nth_group]  = \
                    np.vstack((self.effect.features[nth_group], 
                    np.zeros(self.effect.features[nth_group].shape[1])))
        '''
                  
                    
    def n_feature_groups(self):
        return len(self.context.features)

                  
    def size(self):
        """ Determine the approximate number of elements being used by the
        class and its members. Created to debug an apparently excessive 
        use of memory.
        """
        total = 0
        total += self.context.size()
        total += self.cause.size()
        total += self.effect.size()
        total += self.count.size
        total += self.reward_value.size
        total += self.goal_value.size
        total += self.trace_index.size
        total += self.trace_reward.size

        return total
            
