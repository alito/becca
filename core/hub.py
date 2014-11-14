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
    2) creates new transitions if warranted
    3) updates the amount of priming for each transition
    4) updates the reward estimate for current transitions
    #5) selects a goal and
    #6) declares that goal in the appropriate block.
    """
    def __init__(self, initial_size):
        self.num_cables = initial_size
        # Set constants that adjust the behavior of the hub
        self.INITIAL_REWARD = 0.
        # Features with activities above this threshold are considered 
        # 'active' for the purpose of updating transitions.
        self.ACTIVITY_THRESHOLD = .5
        self.PRIMING_DECAY_RATE = .5
        self.REWARD_LEARNING_RATE = 1e-2
        self.STRENGTH_LEARNING_RATE = 1e-2
        # Keep a history of reward and active features to account for 
        # delayed reward.
        self.TRACE_LENGTH = 10

        # Initialize variables for later use
        self.reward_history = list(np.zeros(self.TRACE_LENGTH))
        self.old_reward = 0.
        feature_shape = (self.num_cables, 1)
        self.cable_activities = np.zeros(feature_shape)
        self.activity_history = [np.zeros(feature_shape)] * (
                self.TRACE_LENGTH + 1)
        # strength, priming, and reward are the three properties of 
        # every transition. In this 2D array representation, each row
        # represents a cause feature, and each column represents 
        # an effect feature. Element [i,j] represents the transition
        # from feature i to feature j.
        transition_shape = (self.num_cables, self.num_cables)
        self.priming = np.zeros(transition_shape)
        self.reward = np.ones(transition_shape) * self.INITIAL_REWARD
        self.strength = np.zeros(transition_shape)

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
        # Determine which features are 'active' for the purpose of 
        # updating transitions.
        active_features = np.zeros(self.cable_activities.shape)
        active_features[np.where(self.cable_activities > 
                                 self.ACTIVITY_THRESHOLD)] = 1.
        # Update the active features history
        self.activity_history.append(active_features)
        self.activity_history.pop(0)
        
        # Update the strength of active transitions
        current_cause = self.activity_history[-1]
        current_effect = self.activity_history[-2]
        self.strength += (current_cause * 
                          (current_effect.T - self.strength) * 
                          self.STRENGTH_LEARNING_RATE)

        # Update the priming
        # Decay the previous priming and add the new priming
        rate = 1. - self.PRIMING_DECAY_RATE
        self.priming = (self.priming * rate + 
                        self.cable_activities * self.strength)
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
            #i = self.TRACE_LENGTH - tau - 1
            #reward_trace += self.reward_history[i] / int(tau + 1)
            reward_trace += self.reward_history[tau] / int(tau + 1)

        # Update the expected reward
        trace_cause = self.activity_history[0]
        trace_effect = self.activity_history[1]
        self.reward += ((reward_trace - self.reward) * 
                       trace_cause) * trace_effect.T * self.REWARD_LEARNING_RATE
        self._display()
        return

    def add_cables(self, num_new_cables):
        """ Add new cables to the hub when new blocks are created """ 
        self.num_cables = self.num_cables + num_new_cables
        features_shape = (self.num_cables, 1)
        transition_shape = (self.num_cables, self.num_cables) 
        self.priming = tools.pad(self.priming, transition_shape)
        self.strength = tools.pad(self.strength, transition_shape)
        self.reward = tools.pad(self.reward, transition_shape,
                                val=self.INITIAL_REWARD)
        self.cable_activities = tools.pad(self.cable_activities, features_shape)

        self.count = tools.pad(self.count, (self.num_cables, self.num_cables))
        # All the cable activities from all the blocks, at the current time
        for index in range(len(self.pre)):
            self.pre[index] = tools.pad(self.pre[index], (self.num_cables, 1))
            self.post[index] = tools.pad(self.post[index], (self.num_cables, 1))

    def _display(self):
        """ Give a visual update of the internal workings of the hub """
        DISPLAY_PERIOD = 10000.
        if np.random.random_sample() < 1. / DISPLAY_PERIOD:

            # Plot reward value
            fig311 = plt.figure(311)
            plt.subplot(2,2,1)
            plt.gray()
            #plt.imshow(np.zeros((32,32)), interpolation='nearest')
            plt.imshow(self.reward.astype(np.float), interpolation='nearest')
            plt.title('reward')
            plt.subplot(2,2,2)
            plt.gray()
            plt.imshow(self.strength, interpolation='nearest')
            plt.title('strength')
            plt.subplot(2,2,3)
            plt.gray()
            plt.imshow(self.priming, interpolation='nearest')
            plt.title('priming')
            #fig311.show()
            #fig311.canvas.draw()
            plt.show()

            #plt.savefig('log/reward_image.png', bbox_inches=0.)
            
