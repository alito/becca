import matplotlib.pyplot as plt
import numpy as np

import tools

class Hub(object):
    """ The hub is the central long term memory and action selection mechanism 
    
    The analogy of the hub and spoke stucture is suggested by 
    the fact that the hub has a connection to each
    of the blocks. In the course of each timestep it 
    1) reads in a copy of the input cable activities to 
        each of the blocks
    2) updates the reward estimate for current transitions
    3) selects a goal and
    4) declares that goal in the appropriate block.
    """
    def __init__(self, initial_size):
        # TODO update comments
        self.num_cables = initial_size
        # Set constants that adjust the behavior of the hub
        self.INITIAL_REWARD = .1
        self.REWARD_LEARNING_RATE = 1e-2
        # Keep a history of reward and active features to account for 
        # delayed reward.
        self.TRACE_LENGTH = 10
        # What fraction of the time are goals selected.
        self.GOAL_FRACTION = .3

        # Initialize variables for later use
        self.reward_history = list(np.zeros(self.TRACE_LENGTH))
        self.old_reward = 0.
        feature_shape = (self.num_cables, 1)
        self.cable_activities = np.zeros(feature_shape)
        self.activity_history = [np.zeros(feature_shape)] * (
                self.TRACE_LENGTH)
        self.action_history = [np.zeros(feature_shape)] * (
                self.TRACE_LENGTH)
        # reward is a property of every transition. 
        # In this 2D array representation, each row
        # represents a feature, and each column represents 
        # a goal (action). Element [i,j] represents the transition
        # from feature i to action j.
        transition_shape = (self.num_cables, self.num_cables)
        self.reward = np.ones(transition_shape) * self.INITIAL_REWARD

    def step(self, blocks, raw_reward):
        """ Advance the hub one step """
        # Gather the cable activities from all the blocks
        cable_index = 0
        block_index = 0
        for block in blocks:
            block_size =  block.cable_activities.size
            self.cable_activities[cable_index: cable_index + block_size] = \
                    block.cable_activities.copy()
            cable_index += block_size 
            block_index += 1
        # Update the reward trace, a decayed sum of recent rewards.
        # Use change in reward, rather than absolute reward.
        raw_reward = float(raw_reward)
        delta_reward = raw_reward - self.old_reward
        self.old_reward = raw_reward
        # Update the reward history
        self.reward_history.append(delta_reward)
        self.reward_history.pop(0)
        # Collapse the reward history into a single value for this time step
        reward_trace = 0.
        for tau in range(self.TRACE_LENGTH):
            # Work from the beginning of the list, from the oldest
            # to the most recent, decaying future values the further
            # they are away from the cause and effect that occurred
            # TRACE_LENGTH time steps ago.
            # Work from the end of the list, from the most recent
            # to the oldest.
            reward_trace += self.reward_history[tau] / float(tau + 1)

        # Update the expected reward
        state = self.activity_history[0]
        action = self.action_history[0]
        if np.where(action != 0.)[0].size:
            self.reward += ((reward_trace - self.reward) * 
                           state * state) * action.T * self.REWARD_LEARNING_RATE
        
        # Only choose goals periodically
        goal = np.zeros(self.num_cables)
        if np.random.random_sample() < self.GOAL_FRACTION:
            # Choose a goal 
            state_weight = self.cable_activities ** 2
            weighted_reward = state_weight * self.reward
            goal_votes = np.sum(weighted_reward, axis=0)
            potential_winners = np.where(goal_votes == np.max(goal_votes))[0] 
            # Break any ties by lottery
            goal_cable = potential_winners[np.random.randint(
                    potential_winners.size)]
            # Figure out which block the goal cable belongs to 
            goal[goal_cable] = 1.
            cable_index = goal_cable
            for block in blocks:
                block_size =  block.hub_cable_goals.size
                if cable_index >= block_size:
                    cable_index -= block_size
                    continue
                else:
                    # Activate the goal
                    block.hub_cable_goals[cable_index] = 1.
        # Update the activity and action history
        self.activity_history.append(np.copy(self.cable_activities))
        self.activity_history.pop(0)
        self.action_history.append(goal)
        self.action_history.pop(0)
        return

    def add_cables(self, num_new_cables):
        """ Add new cables to the hub when new blocks are created """ 
        self.num_cables = self.num_cables + num_new_cables
        features_shape = (self.num_cables, 1)
        transition_shape = (self.num_cables, self.num_cables) 
        self.reward = tools.pad(self.reward, transition_shape,
                                val=self.INITIAL_REWARD)
        self.cable_activities = tools.pad(self.cable_activities, features_shape)

        # All the cable activities from all the blocks, at the current time
        for index in range(len(self.activity_history)):
            self.activity_history[index] = tools.pad(
                    self.activity_history[index], (self.num_cables, 1))
            self.action_history[index] = tools.pad(
                    self.action_history[index], (self.num_cables, 1))

    def _display(self, primed_reward):
        """ Give a visual update of the internal workings of the hub """
        DISPLAY_PERIOD = 10000.
        if np.random.random_sample() < 1. / DISPLAY_PERIOD:

            # Plot reward value
            plt.figure(311)
            plt.subplot(1,2,1)
            plt.gray()
            plt.imshow(self.reward.astype(np.float), interpolation='nearest')
            plt.title('reward')
            plt.subplot(1,2,2)
            plt.gray()
            plt.imshow(primed_reward, interpolation='nearest')
            plt.title('primed reward')
            plt.show()
            #plt.savefig('log/reward_image.png', bbox_inches=0.)
            
