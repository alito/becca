
from utils import *
import numpy as np

class Actor(object):
    """ The reinforcement learner portion of the Becca agent """

    def __init__(self, num_primitives, num_actions, max_num_features):

        self.num_primitives = num_primitives
        self.num_actions = num_actions
        self.max_num_features = max_num_features
        self.action = np.zeros((num_actions,1))

        self.action = np.zeros((num_actions,1))
        self.goal = np.zeros((max_num_features, 1))

        self.AGING_TIME_CONSTANT = 10 ** 6                  # real, large
        self.TRANSITION_UPDATE_RATE = 10 ** -1              # real, 0 < x < 1
        self.GOAL_DECAY_RATE = 0.1                          # real, 0 < x < 1
        self.INITIAL_UNCERTAINTY = 0.5                      # real, 0 < x < 1
        self.MATCH_EXPONENT = 4
        self.REWARD_RANGE_DECAY_RATE = 10 ** -10
        
        self.time_steps = 0
        self.num_transitions = self.max_num_features ** 2
        model_shape = (self.max_num_features, self.num_transitions)
        three_d_eye = np.tile(np.eye(self.max_num_features), (self.max_num_features, 1, 1))
        self.context = np.reshape(three_d_eye.transpose((1,2,0)), model_shape)
        self.cause = np.reshape(three_d_eye.transpose((1,0,2)), model_shape)
        self.effect = np.zeros(model_shape)
        self.effect_uncertainty = np.ones(model_shape) * self.INITIAL_UNCERTAINTY
        
        thin_shape = (1, self.num_transitions)
        self.count = np.zeros(thin_shape)
        self.reward_value = np.zeros(thin_shape)
        self.reward_uncertainty = np.ones(thin_shape) * self.INITIAL_UNCERTAINTY
        self.goal_value = np.zeros(thin_shape)

        state_shape = (self.max_num_features,1)
        self.feature_activity = np.zeros(state_shape)
        self.new_context = np.zeros(state_shape)
        self.new_cause = np.zeros(state_shape)
        self.new_effect = np.zeros(state_shape)
        
        self.reward_min = BIG
        self.reward_max = -BIG
            
        
    def step(self, feature_activity, raw_reward, n_features):
        
        self.feature_activity = feature_activity
        self.num_features = n_features
        self.new_context = self.new_cause.copy()
        self.new_cause = self.feature_activity.copy()
        self.new_effect = self.feature_activity.copy()
        
        """ Modify the reward so that it automatically falls between 0 and 1 """        
        self.reward_min = np.minimum(raw_reward, self.reward_min)
        self.reward_max = np.maximum(raw_reward, self.reward_max)
        spread = self.reward_max - self.reward_min
        self.current_reward = (raw_reward - self.reward_min) / (spread + EPSILON)
        self.reward_min += spread * self.REWARD_RANGE_DECAY_RATE
        self.reward_max -= spread * self.REWARD_RANGE_DECAY_RATE
        
        context_similarities = np.sum((self.new_context * self.context), axis=0)[np.newaxis,:]
        cause_similarities = np.sum((self.new_cause * self.cause), axis=0)[np.newaxis,:]
        transition_similarities = (context_similarities * cause_similarities) ** self.MATCH_EXPONENT

        """ Update the model """
        self.update_model(transition_similarities, self.current_reward)

        """ Decide on an action """
        self.action = self.deliberate(n_features)  
        return self.action


    def update_model(self, transition_similarities, reward):        
        update_rate_raw = transition_similarities * ((1 - self.TRANSITION_UPDATE_RATE) / \
                    (self.count + EPSILON) + self.TRANSITION_UPDATE_RATE)
        update_rate = np.minimum(0.5, update_rate_raw)
        
        reward_difference = np.abs(reward - self.reward_value)
        effect_difference = np.abs(self.new_effect - self.effect)
        self.count += transition_similarities
        self.count -=  np.minimum(1 / (self.AGING_TIME_CONSTANT * self.count + EPSILON), self.count)
        self.reward_value += (reward - self.reward_value) * update_rate
        self.reward_uncertainty += (reward_difference - self.reward_uncertainty) * update_rate
        self.effect += (self.new_effect - self.effect) * update_rate         
        self.effect_uncertainty += (effect_difference - self.effect_uncertainty) * update_rate
        return
   
   
    def deliberate(self, n_features):
        self.goal *= (1 - self.feature_activity) * (1 - self.GOAL_DECAY_RATE)
        context_matches = (np.sum(self.context * 
                                  self.feature_activity, axis=0)[np.newaxis,:]) ** self.MATCH_EXPONENT
        
        estimated_reward_value = self.reward_value + self.reward_uncertainty * \
                    (np.random.random_sample(self.reward_uncertainty.shape) * 2 - 1)
        estimated_reward_value = np.maximum(estimated_reward_value, 0)
        estimated_reward_value = np.minimum(estimated_reward_value, 1)
        reward_value_by_feature = weighted_average(estimated_reward_value, 
                context_matches * self.cause / (self.reward_uncertainty + EPSILON))
        
        goal_jitter = EPSILON * np.random.random_sample(self.effect.shape)
        estimated_effect = self.effect + self.effect_uncertainty * \
                    (np.random.random_sample(self.effect_uncertainty.shape) * 2 - 1)
        estimated_effect = np.maximum(estimated_effect, 0)
        estimated_effect = np.minimum(estimated_effect, 1)
        
        """ Bounded sum over all the goal value amassed in a given transition """
        goal_value_by_transition = map_inf_to_one(np.sum(map_one_to_inf(
                            estimated_effect * (self.goal + goal_jitter)), axis=0)[np.newaxis,:])    
        goal_value_uncertainty_by_transition = map_inf_to_one(np.sum(map_one_to_inf(
                            self.effect_uncertainty * (self.goal + goal_jitter)), axis=0)[np.newaxis,:])
        goal_value_by_feature = weighted_average(goal_value_by_transition, 
                context_matches * self.cause / (goal_value_uncertainty_by_transition + EPSILON))
        
        count_by_feature = weighted_average(self.count, context_matches * self.cause)
        exploration_vote = (1 - self.current_reward) / \
            (n_features * (count_by_feature + 1) * np.random.random_sample(reward_value_by_feature.shape))
        exploration_vote = np.minimum(exploration_vote, 1.)
        exploration_vote[n_features:] = 0.
        
        total_vote = reward_value_by_feature + goal_value_by_feature + exploration_vote
        bounded_total_vote = map_inf_to_one(map_one_to_inf(reward_value_by_feature) + 
                                            map_one_to_inf(goal_value_by_feature) + 
                                            map_one_to_inf(exploration_vote))
        adjusted_vote = total_vote * (1 - self.goal)
        new_goal_feature = np.argmax(adjusted_vote, axis=0)
        self.goal[new_goal_feature, :] = np.maximum(bounded_total_vote[new_goal_feature, :], 
                                                    self.goal[new_goal_feature, :])
        
        """ Strip the actions off the goals to make the current set of actions """
        self.goal[self.num_primitives: self.num_primitives + self.num_actions,:] = \
                np.sign(self.goal[self.num_primitives: self.num_primitives + self.num_actions,:])
        action = np.copy(self.goal[self.num_primitives: self.num_primitives + self.num_actions,:])
        return action

                 
    def visualize(self, save_eps=True):
        import viz_utils
        viz_utils.visualize_model(self, self.num_primitives, self.num_actions, 10)
        viz_utils.force_redraw()
        return