import logging

import numpy as np

import utils

class Planner(object):
    """
    Planner
    """
    def __init__(self, num_actions):
        self.EXPLORATION_FRACTION = 0.2     # real, 0 < x < 1
        self.OBSERVATION_FRACTION = 0.5    # real, 0 < x < 1

        self.act = False
        self.action = np.zeros(num_actions)


    def explore(self):
        """
        Forms a planner plan
        """
        
        self.act = True
        
        # Exploratory commands are only generated at the basic command feature
        # level (k = 3). Features and higher-level commands are not excited.

        num_actions = np.size(self.action)
        #handles motor commands as all-or-none
        self.action = np.zeros(num_actions)

        # old code: only one action element active
        self.action[np.random.random_integers(0, num_actions-1)] = 1

        #     % new code
        #     N = 1; %adjust this<97>typical number of ones per exploratory action
        #     frac = (length( planner.plan{k}) + N) / length( planner.plan{k});
        #     planner.plan{k}( find( rand( size( planner.plan{k})) * frac > 1)) = 1;


    def select_action(self, model, current_state):
        
        """
        Chooses a deliberative action based on the current feature activities.
        Finds the weighted expected reward for the actions across all model 
        entries. Then executes the actions with a magnitude that is a function 
        of the expected reward associated with each.
        
        It's a low-level all-to-all planner, capable of executing many plans in
        parallel. Even conflicting ones.

        """

        eps = np.finfo(np.double).eps 
        
        # When goals are implemented, combine the reward value 
        # associated with each model
        # entry with the goal value associated with it.
        effect_values = model.reward_value[:model.n_transitions]

        # Create a shorthand for the variables to keep the code readable.
        model_actions = model.cause[2][:, :model.n_transitions]
        count_weight = np.log(model.count[:model.n_transitions] + 1)
        value = effect_values
        similarity = utils.similarity(current_state, model.cause, model.n_transitions)

        # The reactive action is a weighted average of all actions. Actions 
        # that are expected to result in a high value state and actions that are
        # similar to the current state are weighted more heavily. All actions 
        # computed in this way will be <= 1.
        weights = count_weight * value * similarity

        # debug
        # Picks closest weight only.
        max_indices = np.argmax(weights)
        max_index = max_indices[np.random.random_integers(0, len(max_indices)-1)]

        weights = np.zeros(np.size(weights))
        weights[max_index] = 1

        positive_weights = weights > 0
        negative_weights = weights < 0

        sizer = np.ones(model_actions.shape[1])
        weight_mat = np.dot(sizer, weights)
        action_positive = np.sum(model_actions[:,positive_weights] * weight_mat[:,positive_weights], 1) / \
                    (np.sum(weight_mat[:,positive_weights], 1) + eps)
        action_negative = np.sum( model_actions[:,negative_weights] * weight_mat[:,negative_weights], 1) / \
                    (np.sum( weight_mat[:,negative_weights], 1) + eps)

        action = action_positive - action_negative

        # Sets a floor for action magnitudes. Negative actions are not defined.
        action = np.maximum(action, 0)

        return action



    def deliberate(self, agent):
        """
        Deliberates, choosing goals based on the current working memory.
        It finds the transition that is best suited based on:
          1) similarity
          2) count
          3) expected reward
          4) not already having a strong goal that matches the cause
        It chooses cause of the winning transition as the goal for the timestep.
        """
        
        model = agent.model

        # decays goals both by a fixed fraction and by the amount that the feature
        # is currently active. Experiencing a goal feature causes the goal to be
        # achieved, and the passage of time allows the goal value to fade.
        for index in range(agent.num_groups):
            if np.size(agent.goal[index]):
                agent.goal[index] *= (1 - agent.feature_activity[index])
                agent.goal[index] *= (1 - agent.GOAL_DECAY_RATE)

        # Calculates the value associated with each effect
        goal_value = np.zeros(model.n_transitions)
        for index in range (1, agent.num_groups):
            # [newaxis] needed to make it 2D
            splashed = np.dot(agent.goal[index][np.newaxis].transpose(), np.ones((1, model.n_transitions)))
            goal_value += np.sum(model.effect[index][:, :model.n_transitions] * splashed, 0)


        # Sets maximum goal value to 1.
        goal_value = np.minimum(goal_value, 1)
        reward_value = model.reward_value[:model.n_transitions] - agent.reward

        # Combines goal-based and reward-based value, bounded by one.
        # The result is a value for each transition
        value = utils.bounded_sum(goal_value, reward_value)

        # Each transition's count and its similarity to the working memory also
        # factor in to its vote
        #count_weight = utils.sigmoid(np.log(model.count[:model.n_transitions] + 1) / 3)

        similarity = utils.similarity(agent.working_memory, model.context, model.n_transitions)

        # TODO: Raise similarity by some power to focus on more similar transitions?

        # TODO: Introduce the notion of reliability? That is, when a transition is
        # used for planning, whether the intended plan is executed?

        # TODO: Address the lottery problem? (Reliability is one possible solution)
        # This is the problem in which less common, but rewarding, transitions are
        # selected over more common transitions. The distinguishing factor should
        # not be the count, but the reliability of the transition. Fixing the
        # lottery problem may make BECCA less human-trainable.

        # TODO: Add recency? This is likely to be useful when rewards change over
        # time. 

        # Scales the vote by the distance between the cause and any current goals
        goal_distance = np.zeros(model.n_transitions)
        for index in range(1,agent.num_groups):            
            goal_distance_group = np.max (model.cause[index][:,:model.n_transitions] - \
                                              np.dot(agent.goal[index][np.newaxis].transpose(),
                                                     np.ones((1, model.n_transitions))))
            goal_distance = np.maximum(goal_distance, goal_distance_group)


        # ### DEBUG
        # # Scales the vote by the distance between the cause and any currently
        # # active features
        # feature_dist = zeros( 1, model.n_transitions)
        # for k = 2:agent.num_groups,
        # #     feature_dist_group = max (model.cause[k](:,1:model.n_transitions) - ...
        # #         agent.feature_activity[k] * ones( 1, model.n_transitions))
        #     feature_dist_group = max (model.cause[k](:,1:model.n_transitions) - ...
        #         agent.attended_feature[k] * ones( 1, model.n_transitions))
        #     feature_dist = max( feature_dist, feature_dist_group)
        # end
        # ###

        # DEBUG
        transition_vote = value * similarity * goal_distance ** 4
        # transition_vote = count_weight * value * similarity * goal_distance ** 4
        # transition_vote = count_weight * value * similarity .* goal_distance ** 4 * feature_dist ** 4
        
        max_transition_index = np.argmax(transition_vote)

        if transition_vote[max_transition_index] > 0:
            for index in range(1,agent.num_groups):
                nonzeros = model.cause[index][:, max_transition_index].ravel().nonzero()[0]
                if np.size(nonzeros):
                    max_goal_feature = nonzeros[0]
                    max_goal_group = index
                    agent.goal[max_goal_group][max_goal_feature] = \
                        utils.bounded_sum(agent.goal[max_goal_group][max_goal_feature],
                        value[max_transition_index] * agent.STEP_DISCOUNT)



                    # print max_goal_group
                    # print max_goal_feature
                    # print agent.goal[max_goal_group].transpose()
                    
        # primitive action goals can be fulfilled immediately
        self.action = np.zeros(np.size(self.action))
        if np.size((agent.goal[2] > 0).nonzero()):
            self.act = True

            self.action[agent.goal[2] > 0] = 1
            agent.goal[2] = np.zeros(np.size( agent.goal[2]))


    def step(self, agent):
        # Reactively chooses an action.
        # TODO: make reactive actions habit based, not reward based
        # also make reactive actions general
        # reactive_action = self.select_action(self.model, self.feature_activity)

        # only acts deliberately on a fraction of the time steps
        if np.random.random_sample() > self.OBSERVATION_FRACTION:
            # occasionally explores when making a deliberate action.
            # Sets self.planner.action
            if np.random.random_sample() < self.EXPLORATION_FRACTION:
                logging.debug('EXPLORING')
                self.explore()
            else:
                logging.debug('DELIBERATING')
                # Deliberately choose features as goals, in addition to actions.
                self.deliberate(agent)

        else:
            self.action = np.zeros( self.action.shape[0])
        
        return self.action
            
