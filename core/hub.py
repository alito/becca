""" the Hub class """
import matplotlib.pyplot as plt
import numpy as np
import tools

class Hub(object):
    """ 
    The central long term memory and action selection mechanism 
    
    The analogy of the hub and spoke stucture is suggested by 
    the fact that the hub has a connection to each
    of the gearboxes. In the course of each timestep it 
    1) reads in a copy of the input cable activities to each of the gearboxes
    2) updates the reward estimate for current transitions
    3) selects a goal and
    4) declares that goal in the appropriate gearbox.
    """
    def __init__(self, initial_size):
        self.num_cables = initial_size
        # Set constants that adjust the behavior of the hub
        self.INITIAL_REWARD = .5
        self.REWARD_LEARNING_RATE = .1
        # Keep a history of reward and active features to account for 
        # delayed reward.
        self.TRACE_LENGTH = 10

        # Initialize variables for later use
        self.reward_history = list(np.zeros(self.TRACE_LENGTH))
        self.old_reward = 0.
        feature_shape = (self.num_cables, 1)
        self.activity_history = [np.zeros(feature_shape)] * (self.TRACE_LENGTH)
        self.action_history = [np.zeros(feature_shape)] * (self.TRACE_LENGTH)
        # reward is a property of every transition. 
        # In this 2D array representation, each row
        # represents a feature, and each column represents 
        # a goal (action). Element [i,j] represents the transition
        # from feature i to action j.
        transition_shape = (self.num_cables, self.num_cables)
        self.reward = np.ones(transition_shape) * self.INITIAL_REWARD

    def step(self, feature_activities, raw_reward):
        """ Advance the hub one step """
        # Update the reward trace, a decayed sum of recent rewards.
        self.reward_history.append(raw_reward)
        self.reward_history.pop(0)
        # Collapse the reward history into a single value for this time step
        reward_trace = 0.
        for tau in range(self.TRACE_LENGTH):
            # Work from the end of the list, from the most recent
            # to the oldest, decaying future values the further
            # they are away from the cause and effect that occurred
            # TRACE_LENGTH time steps ago.
            reward_trace += self.reward_history[tau] / float(tau + 1)

        # Update the expected reward
        state = self.activity_history[0]
        action = self.action_history[0]
        # To avoid unnecessary computation, check whether there has been an 
        # action take before bothering to update the reward.
        if np.where(action != 0.)[0].size:
            self.reward += (((reward_trace - self.reward) * 
                            state * state) * action.T * 
                            self.REWARD_LEARNING_RATE)
        
        # Choose a goal at every timestep
        goal = np.zeros((self.num_cables, 1))
        state_weight = feature_activities ** 2
        weighted_reward = state_weight * self.reward
        expected_reward = np.sum(weighted_reward, axis=0) / np.sum(state_weight)
        best_reward = np.max(expected_reward)
        potential_winners = np.where(expected_reward == best_reward)[0] 
        # Break any ties by lottery
        if potential_winners.size > 0:
            goal_cable = potential_winners[np.random.randint(
                    potential_winners.size)]
        else:
            goal_cable = np.random.randint(goal.size)

        return goal_cable, best_reward
       
    def update(self, feature_activities, issued_goal_index): 
        """ Assign the goal to train on, based on the goal that was issued """
        goal = np.zeros((self.num_cables, 1))
        if issued_goal_index is not None:
            goal[issued_goal_index] = 1.
        # Update the activity and action history
        self.activity_history.append(np.copy(feature_activities))
        self.activity_history.pop(0)
        self.action_history.append(goal)
        self.action_history.pop(0)
        #self.visualize()
        return

    def add_cables(self, num_new_cables):
        """ Add new cables to the hub when new gearboxes are created """ 
        self.num_cables = self.num_cables + num_new_cables
        features_shape = (self.num_cables, 1)
        transition_shape = (self.num_cables, self.num_cables) 
        self.reward = tools.pad(self.reward, transition_shape,
                                val=self.INITIAL_REWARD)

        for index in range(len(self.activity_history)):
            self.activity_history[index] = tools.pad(
                    self.activity_history[index], features_shape)
            self.action_history[index] = tools.pad(
                    self.action_history[index], features_shape)

    def visualize(self):
        """ Give a visual update of the internal workings of the hub """
        # Plot reward value
        plt.figure(311)
        plt.subplot(1,2,1)
        plt.gray()
        plt.imshow(self.reward.astype(np.float), interpolation='nearest')
        plt.title('reward')
        #plt.subplot(1,2,2)
        #plt.gray()
        #plt.imshow(primed_reward, interpolation='nearest')
        #plt.title('primed reward')
        plt.show()
        #plt.savefig('log/reward_image.png', bbox_inches=0.)
            
