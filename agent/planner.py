
import numpy as np
import utils

class Planner(object):

    def __init__(self, num_primitives, num_actions, max_num_features):
        
        self.num_primitives = num_primitives
        self.num_actions = num_actions
        self.action = np.zeros((num_actions,1))
        
        self.goal = np.zeros((max_num_features, 1))
        self.debug = False


    def step(self, model, attended_feature, n_features):
      
        #print 'self.planner.goal before', self.goal[np.nonzero(self.goal)[0],:].ravel(), \
        #                        np.nonzero(self.goal)[0].ravel()
        self.goal *= 1 - model.feature_activity
        #self.goal -= np.minimum(model.feature_activity, self.goal)
        self.GOAL_DECAY_RATE = 0.1
        self.goal *= 1 - self.GOAL_DECAY_RATE

        self.debug = False
        if np.random.random_sample() < 0.01:
            self.debug = True
            
        if self.debug:
            print 'new iteration ======================================================================='
            print 'model.feature_activity', model.feature_activity[:n_features,:].ravel()
            print 'self.goal', self.goal[np.nonzero(self.goal)[0],:].ravel(), \
                                   np.nonzero(self.goal)[0].ravel()
        

        self.deliberate(model, attended_feature, n_features)  
        
        #reaction = self.react(model)
        
        #self.goal = reaction + excitation - inhibition
        #self.goal = np.maximum(self.goal, 0.)
        #self.goal = np.minimum(self.goal, 1.)
        
        #print reaction.ravel(), 'reaction'
        #print excitation.ravel(), 'excitation'
        #print inhibition.ravel(), 'inhibition'
        #print self.goal.ravel(), 'goal'
        
        #self.planner_salience = np.minimum(excitation + inhibition, 1.)


        """ Separate action goals from the rest of the goal """
        self.action = np.sign(self.goal[self.num_primitives: self.num_primitives + self.num_actions,:])
        #self.action = np.maximum(self.action, 0.)

        #debug
        #self.goal = np.zeros(self.goal.shape)
        
        return self.action
            
            
    def deliberate(self, model, attended_feature, n_features):
        
        context_matches = np.sum(model.context * attended_feature, axis=0)[np.newaxis,:]
        
        match_indices = np.nonzero(context_matches)[1]
        """ calculated value by transition """
        
        penalized_reward_value = np.maximum(model.reward_value - model.reward_uncertainty, 0)
        reward_value_by_feature = utils.weighted_average(penalized_reward_value, 
                                context_matches * model.cause / (model.reward_uncertainty + utils.EPSILON))

        goal_indices = np.argmax((model.effect - model.effect_uncertainty) * self.goal, axis=0)
        nonzero_goal_indices = np.nonzero(goal_indices)[0]
        
        transition_indices = np.arange(model.effect.shape[1])
        
        goal_value_by_transition = model.effect[goal_indices,transition_indices] - \
                                   model.effect_uncertainty[goal_indices,transition_indices]
        goal_value_uncertainty_by_transition = model.effect_uncertainty[goal_indices,transition_indices]
        penalized_goal_value = np.maximum(goal_value_by_transition - 
                                          goal_value_uncertainty_by_transition, 0)
        goal_value_by_feature = utils.weighted_average(penalized_goal_value, 
                        context_matches * model.cause / (goal_value_uncertainty_by_transition + utils.EPSILON))
        #goal_value = utils.map_inf_to_one(np.sum(model.effect * utils.map_one_to_inf(self.goal), axis=0))
        #goal_value_uncertainty = np.sum(model.effect_uncertainty * np.abs(self.goal), axis=0) / \
        #                         np.sqrt(np.sum(np.abs(self.goal), axis=0) + utils.EPSILON)
                                 
        #penalized_goal_value = np.maximum(goal_value - goal_value_uncertainty, 0)
        #goal_value_by_feature = utils.weighted_average(penalized_goal_value, 
        #                        context_matches * model.cause / (goal_value_uncertainty + utils.EPSILON))
                                 
        '''value_total = model.reward_value - model.reward_uncertainty + goal_value - goal_value_uncertainty
        value_total = np.maximum(value_total, 0)
        value_uncertainty = model.reward_uncertainty + goal_value_uncertainty
        expected_reward_value = utils.weighted_average(value_total, 
                                context_matches * model.cause / (value_uncertainty + utils.EPSILON))
        '''
        value_by_feature = reward_value_by_feature + goal_value_by_feature
        
        count_by_feature = utils.weighted_average(model.count, context_matches * model.cause)
        self.EXPLORATION_FACTOR = 1.
        exploration_vote = self.EXPLORATION_FACTOR / (count_by_feature + 1)
        exploration_vote[n_features:] = 0.
        
        total_vote = exploration_vote + value_by_feature
        adjusted_vote = total_vote * (1 - self.goal)
        adjusted_vote_power = adjusted_vote ** n_features
        cumulative_vote = np.cumsum(adjusted_vote_power,axis=0) / np.sum(adjusted_vote_power, axis=0)
        new_goal_feature = np.nonzero(np.random.random_sample() < cumulative_vote)[0][0]
        self.goal[new_goal_feature, :] += adjusted_vote[new_goal_feature, :]
        self.goal[new_goal_feature, :] = np.minimum(self.goal[new_goal_feature, :], 1)
        
        #print 'adjusted_vote', adjusted_vote.ravel()
        
        if self.debug:
            #if np.random.random_sample() < 0.0:
            print 'context_matches', match_indices.ravel()
            print 'model.context', model.context[:n_features,match_indices]
            print 'model.cause', model.cause[:n_features,match_indices]        
            print 'model.effect', model.effect[:n_features,match_indices]        
            print 'model.effect_uncertainty', model.effect_uncertainty[:n_features,match_indices]   
            print 'model.reward',  model.reward_value[:,match_indices]     
            print 'model.reward_uncertainty',  model.reward_uncertainty[:,match_indices]     
            print 'model.feature_activity', model.feature_activity[:n_features,:].ravel()
            print 'nonzero goal_indices', nonzero_goal_indices
            #print 'goal_value_by_transition', goal_value_by_transition[nonzero_goal_indices]
            #print 'goal_value_uncertainty_by_transition', goal_value_uncertainty_by_transition[nonzero_goal_indices]
            #print 'penalized_goal_value', penalized_goal_value[nonzero_goal_indices]
        
            print 'goal_value_by_transition', goal_value_by_transition[:,match_indices].ravel()
            print 'goal_value_uncertainty_by_transition', goal_value_uncertainty_by_transition[:,match_indices].ravel()
            print 'reward_value_by_feature', reward_value_by_feature.ravel()
            print 'goal_value_by_feature', goal_value_by_feature.ravel()
            print 'value_by_feature', value_by_feature.ravel()
            print 'model.current_reward', model.current_reward
            
            print 'count_by_feature', count_by_feature.ravel()
            print 'exploration_vote', exploration_vote.ravel()
            print 'total_vote', total_vote.ravel()
            print 'adjusted_vote', adjusted_vote.ravel()
            print 'adjusted_vote_power', adjusted_vote_power.ravel()
            print 'cumulative_vote', cumulative_vote.ravel()
            print 'adjusted_vote[new_goal_feature, :]', adjusted_vote[new_goal_feature, :]
            print 'new_goal_feature', new_goal_feature
            print 'total_vote[new_goal_feature, :]', total_vote[new_goal_feature, :]
                        
            print 'self.goal', self.goal.ravel()

        return
    

    def react(self, model):
        '''predicted_action = model.prediction[self.num_primitives: 
                                             self.num_primitives + self.num_actions,:]
        prediction_uncertainty = model.prediction_uncertainty[self.num_primitives: 
                                             self.num_primitives + self.num_actions,:]
        #reaction_threshold =  np.maximum(predicted_action ** 0.5 - prediction_uncertainty, 0)
        #reaction_threshold =  predicted_action
        
        #reaction = np.zeros(reaction_threshold.shape)
        #reaction[np.nonzero(np.random.random_sample(reaction.shape) < reaction_threshold)] = 1.0
            
        reaction =  predicted_action - prediction_uncertainty
        '''
        prediction, prediction_uncertainty = model.predict_next_step()
        reaction = prediction - prediction_uncertainty
        """ TODO: try this instead ? """
        #reaction = prediction
        
        if np.random.random_sample() < 0: 
            #if np.max(reaction.ravel()) > 0.01:
            print 'prediction', predicted_action.ravel(), ' uncertainty', prediction_uncertainty.ravel()
            #print 'reaction_threshold', reaction_threshold.ravel()
            print 'reaction', reaction.ravel()
        
        return reaction
