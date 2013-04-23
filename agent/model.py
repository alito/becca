import utils as ut

import numpy as np

class Model(object):
    """
    Track the tendency of one input to follow another

    The model is the basis for prediction in the agent. 
    It incrementally estimates the likelihood that a feature
    will occur (the effect), given the set of features that have recently
    occurred (the cause). Each cause-effect is represented by a transition.
    Each transition also tracks the expected reward, the uncertainty in
    the estimates of the effect and reward, and a count of how many
    times the transition has occurred.
    
    New inputs are used to update the model at each timestep during
    upward processing. During downward processing, the descending goals
    are processed through the model and, combined with the model's own
    reward estimates, used to choose lower level goals. 
    """
    def __init__(self, max_num_features, name):
        """ Initialize the model, preallocating all data structures """
        self.max_num_features = max_num_features
        self.name = name
        self.AGING_TIME_CONSTANT = 10 ** 6 # real, large
        self.TRANSITION_UPDATE_RATE = 10 ** -1 # real, 0 < x < 1
        self.CAUSE_DECAY_RATE = .33 # real, 0 < x < 1
        self.GOAL_DECAY_RATE = 0.1 # real, 0 < x < 1
        self.INITIAL_UNCERTAINTY = 0.5 # real, 0 < x < 1
        self.MATCH_EXPONENT = 1 # real, 0 < x 
        
        self.time_steps = 0
        model_shape = (max_num_features, max_num_features)        
        self.count = np.zeros(model_shape)
        self.cause_count = np.zeros(model_shape)
        self.expected_effect = np.zeros(model_shape)
        self.effect_uncertainty = np.zeros(model_shape)
        self.reward_value = np.zeros(model_shape)
        self.reward_uncertainty = (np.ones(model_shape) *
				   self.INITIAL_UNCERTAINTY)
        state_shape = (max_num_features,1)
        self.new_cause = np.zeros(state_shape)
        self.new_effect = np.zeros(state_shape)
        self.num_feature_inputs = 0
             
        self.goal = np.zeros((max_num_features, 1))
        self.surprise = np.ones((max_num_features, 1))

    def update(self, feature_input, reward):        
        """ Train the model using the current feature_input and reward """
        self.num_feature_inputs = np.maximum(self.num_feature_inputs, 
                                             feature_input.size)
        # Pad the incoming feature_input array out to its full size 
        feature_input = ut.pad(feature_input, (self.max_num_features, 0))
        self.current_reward = reward
        # The cause is a weighted sum of previous inputs, with the most
        # recent inputs being weighted the highest
        self.new_cause = ut.bounded_sum([
                self.new_effect, self.new_cause * (1 - self.CAUSE_DECAY_RATE)])
        self.new_effect = feature_input
        transition_activities = ((self.new_cause * self.new_effect.T) **
                                 self.MATCH_EXPONENT)
        update_rate_raw = (transition_activities * 
                           ((1 - self.TRANSITION_UPDATE_RATE) / 
                            (self.count + ut.EPSILON) + 
                            self.TRANSITION_UPDATE_RATE))
        update_rate = np.minimum(0.5, update_rate_raw)
        self.count += transition_activities
        self.count -= 1 / (self.AGING_TIME_CONSTANT * self.count + 
                           ut.EPSILON)
        self.count = np.maximum(self.count, 0)
        reward_difference = np.abs(reward - self.reward_value)
        self.reward_value += (reward - self.reward_value) * update_rate
        self.reward_uncertainty += (reward_difference - 
                                    self.reward_uncertainty) * update_rate
        update_rate_raw_effect = (self.new_cause * 
                                  ((1 - self.TRANSITION_UPDATE_RATE) / 
                                  (self.cause_count + ut.EPSILON) + 
		                  self.TRANSITION_UPDATE_RATE)) 
        update_rate_effect = np.minimum(0.5, update_rate_raw_effect)
        self.cause_count += self.new_cause
        self.cause_count -= 1 / (self.AGING_TIME_CONSTANT * self.cause_count +
                                   ut.EPSILON)
        self.cause_count = np.maximum(self.cause_count, 0)
        effect_difference = np.abs(self.new_cause * self.new_effect.T - 
				   self.expected_effect)
        self.expected_effect += (self.new_cause * self.new_effect.T - 
				 self.expected_effect) * update_rate_effect
        self.effect_uncertainty += (effect_difference - 
                                self.effect_uncertainty) * update_rate_effect 
        # Reaction is the expected effect, turned into a goal
        self.reaction = ut.weighted_average(self.expected_effect, 
                                            self.new_cause)
        # Surprise is the difference between the expected effect and
        # the actual one
        self.surprise = ut.weighted_average(
                np.abs((self.new_effect.T - self.expected_effect) / 
		       (self.effect_uncertainty + ut.EPSILON)), 
		self.new_cause / (self.effect_uncertainty + ut.EPSILON))
        # Reshape transition activities into a single column
        return transition_activities.ravel()[:,np.newaxis]
   
    def deliberate(self, goal_value_by_transition):
        """ Choose goals deliberatively, based on goal and reward value """
        # Maintain the internal goal set
        goal_fulfillment = 1 - self.new_effect
        goal_decay = 1 - self.GOAL_DECAY_RATE
        self.goal *= goal_fulfillment * goal_decay

        similarity = np.tile(self.new_effect, (1,self.new_effect.size))
        reward_noise = (np.random.random_sample(
                self.reward_uncertainty.shape)* 2 - 1)
        estimated_reward_value = (self.reward_value - self.current_reward + 
                                  self.reward_uncertainty * reward_noise)
        estimated_reward_value = np.maximum(estimated_reward_value, 0)
        estimated_reward_value = np.minimum(estimated_reward_value, 1)
        reward_value_by_feature = ut.weighted_average(
                estimated_reward_value, 
                similarity / (self.reward_uncertainty + ut.EPSILON))
        reward_value_by_feature[self.num_feature_inputs:] = 0. 
        # Reshape goal_value_by_transition back into a square array 
        goal_value_by_transition = np.reshape(goal_value_by_transition, 
                                              (self.goal.size, -1))
        # Bounded sum of the goal values from above over all transitions 
        goal_value_by_feature = ut.bounded_sum(goal_value_by_transition.T * 
                                               similarity)
        count_by_feature = ut.weighted_average(self.count, similarity)
        exploration_vote = ((1 - self.current_reward) / 
                (self.num_feature_inputs * (count_by_feature + 1) * 
                 np.random.random_sample(count_by_feature.shape) + ut.EPSILON))
        exploration_vote = np.minimum(exploration_vote, 1.)
        exploration_vote[self.num_feature_inputs:] = 0.
        #exploration_vote = np.zeros(reward_value_by_feature.shape)
        # debug
        include_goals = True
        if include_goals:
            total_vote = (reward_value_by_feature + goal_value_by_feature +
                            exploration_vote)
            vote = ut.bounded_sum([reward_value_by_feature, 
                                goal_value_by_feature, exploration_vote])
        else:
            total_vote = reward_value_by_feature + exploration_vote
            vote = ut.bounded_sum([reward_value_by_feature, 
                                  exploration_vote])
        self.goal = np.maximum( vote, self.goal)
        return vote[:self.num_feature_inputs]

    def get_goal(self):
        return self.goal[:self.num_feature_inputs]

    def get_reaction(self):
        return self.reaction[:self.num_feature_inputs]

    def get_surprise(self):
        return self.surprise[:self.num_feature_inputs]

    def get_projection(self, map_projection):
        """ Find the projection from transition activities to inputs """
        num_inputs = self.reward_value.shape[0]
        projection = np.zeros((num_inputs,2))
        transitions = np.reshape(map_projection, (num_inputs,num_inputs))
        projection[:,0] = np.sign(np.max(transitions, axis=1))
        projection[:,1] = np.sign(np.max(transitions, axis=0))
        return projection
    
    def visualize(self, save_eps=True):
        """ Show the internal state of the model in a pictorial format """
        import utils
        utils.visualize_array(self.reward_value, 
                                  label=self.name + '_reward')
        #utils.visualize_array(self.reward_uncertainty, 
        #                          label=self.name + '_reward_uncertainty')
        utils.visualize_array(np.log(self.count + 1.), 
                                  label=self.name + '_count')
        #utils.visualize_model(self, self.num_primitives, 
        #                          self.num_actions, 10)
        return
