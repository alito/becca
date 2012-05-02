
import numpy as np
import utils

class Planner(object):

    def __init__(self, num_actions):
        
        """ The approximate fraction of time steps on which the 
        planner makes a random, exploratory plan.
        """
        self.EXPLORATION_FRACTION = 0.2     # real, 0 < x < 1
        
        """ The approximate fraction of time steps on which the 
        planner intentionally does nothing so that the model can observe.
        """
        self.OBSERVATION_FRACTION = 0.5    # real, 0 < x < 1

        self.deliberately_acted = False
        self.action = np.zeros(num_actions)


    def step(self, model, working_memory):
        """ Choose an action at each time step """
        
        """ First, choose a reactive action """
        """ TODO: make reactive actions habit based, not reward based
        also make reactive actions general """
        # debug
        # reactive_action = self.select_action(self.model, self.feature_activity)

        deliberately_acted = False
        
        """ Second, choose a deliberate action (or non-action) """
        """ Only act deliberately on a fraction of the time steps """
        if np.random.random_sample() > self.OBSERVATION_FRACTION:
            
            """ Occasionally explore when making a deliberate action """
            if np.random.random_sample() < self.EXPLORATION_FRACTION:
                print('EXPLORING')
                self.action = self.explore()
                            
                """ Attend to any deliberate actions """
                deliberately_acted = True

            else:
                print('DELIBERATING')
                """ The rest of the time, deliberately choose an action.
                Choose features as goals, in addition to actions.
                """
                (self.action, goal) = self.deliberate(model, working_memory)
                
                """ Pass goal to model """ 
                model.update_goal(goal)
                
                if np.count_nonzero(self.action) > 0:
                    """ Attend to any deliberate actions """
                    deliberately_acted = True
        
                #debug
                print self.action                

        else:
            self.action = np.zeros( self.action.shape[0])
        
        return self.action, deliberately_acted
            

    def explore(self):
        """ Forms a random, exploratory plan """
        """ TODO: form and execute multi-step plans, rather than single-step
        random actions.
        """        
        """ Exploratory commands are only generated at the basic 
        action feature level. Features and higher-level commands 
        are not excited.
        """
        num_actions = np.size(self.action)
        
        """ Set all elements of the action to zero """
        action = np.zeros(num_actions)

        """ When exploring, randomly pick one action element to be active """
        action[np.random.random_integers(0, num_actions-1)] = 1
        
        return action

    '''
    def select_action(self, model, current_state):
        """
        Choose a reactive action based on the current feature activities.
        This is analogous to automatic actions
        Finds the weighted expected reward for the actions across all model 
        transitions. Then executes the actions with a magnitude that 
        is a function of the expected reward associated with each.
        
        It's a low-level all-to-all planner, capable of executing many plans in
        parallel. Even conflicting ones.
        """

        eps = np.finfo(np.double).eps 
        
        """ When goals are implemented, combine the reward value 
        associated with each model
        entry with the goal value associated with it. 
        """
        effect_values = model.reward_value[:model.n_transitions]

        """ Create a shorthand for the variables to keep the code readable """
        model_actions = model.cause.actions[:, :model.n_transitions]
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
        '''

    def deliberate(self, model, working_memory):
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
        value = utils.bounded_sum(model.goal_value[:model.n_transitions], 
                                  model.reward_value[:model.n_transitions])

        """ Each transition's count and its similarity to the working memory 
        also factor in to its vote.
        """
        #count_weight = utils.sigmoid(np.log(model.count
        #        [:model.n_transitions] + 1) / 3)

        """ TODO: query model, so I don't have to probe 
        its members directly. 
        """
        similarity = utils.similarity(working_memory, model.context, 
                                      model.n_transitions)

        """ TODO: Raise similarity by some power to focus on more 
        similar transitions?
        """
        """ TODO: Introduce the notion of reliability? That is, when 
        a transition is used for planning, whether the intended plan 
        is executed?
        """
        """ TODO: Address the lottery problem? (Reliability is 
        one possible solution)
        This is the problem in which less common, but rewarding, 
        transitions are selected over more common transitions. The 
        distinguishing factor should not be the count, but the 
        reliability of the transition. Fixing the
        lottery problem may make BECCA less human-trainable.
        """
        """ TODO: Add recency? This is likely to be useful when 
        rewards change over time. 
        """
        
        transition_vote = value * similarity
        max_transition_index = np.argmax(transition_vote)

        goal = working_memory.zeros_like()
        goal.primitives = model.cause.primitives[:, max_transition_index]
        goal.actions = model.cause.actions[:, max_transition_index]
        for group_index in range(model.n_feature_groups()):
            goal.features[group_index] = model.cause.features[group_index] \
                                                [:, max_transition_index]
                
        """ Separate action goals from the rest of the goal """
        action = np.zeros(np.size(self.action))
        if np.size((goal.actions > 0).nonzero()):
            self.deliberately_acted = True

            action[goal.actions > 0] = 1
            goal.actions = np.zeros(np.size(goal.actions))

        return action, goal