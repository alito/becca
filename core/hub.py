import matplotlib.pyplot as plt
import numpy as np

import tools

class Hub(object):
    """ The hub is the central action selection mechanism 
    
    The analogy of the hub and spoke stucture is suggested by 
    the fact that the hub has a connection to each
    of the blocks. In the course of each timestep it 
    1) reads in a copy of the input cable activities to 
        each of the blocks
    2) updates its reward distribution estimate for all 
        the goals it could select
    3) selects a goal and
    4) declares that goal in the appropriate block.
    """
    def __init__(self, initial_size):
        self.num_cables = initial_size 

        # Set constants that adjust the behavior of the hub
        self.INITIAL_REWARD = 1.0
        self.UPDATE_RATE = 10 ** -2.
        self.REWARD_DECAY_RATE = .3
        self.FORGETTING_RATE = 10 ** -5
        self.TRACE_LENGTH = 10
        self.EXPLORATION = .1
        
        # Initialize variables for later use
        self.reward_min = tools.BIG
        self.reward_max = -tools.BIG
        self.old_reward = 0.
        self.count = np.zeros((self.num_cables, self.num_cables))
        self.reward_trace = [0.] * self.TRACE_LENGTH
        self.expected_reward = (np.ones((self.num_cables, self.num_cables)) *
                             self.INITIAL_REWARD)
        self.cable_activities = np.zeros((self.num_cables, 1))
        # pre represents the feature and sensor activities at a given
        # time step.
        # post represents the goal or action that was taken following. 
        self.pre = [np.zeros((self.num_cables, 1))] * (self.TRACE_LENGTH) 
        self.post = [np.zeros((self.num_cables, 1))] * (self.TRACE_LENGTH)
        return
    
    def step(self, blocks, unscaled_reward):
        """ Advance the hub one step:
        1. Comb tower of blocks, collecting cable activities from each
        2. Update all-to-all reward model
        3. Select a goal
        4. Modify the goal in the block
        """
        # Adapt the reward so that it falls between -1 and 1 
        self.reward_min = np.minimum(unscaled_reward, self.reward_min)
        self.reward_max = np.maximum(unscaled_reward, self.reward_max)
        spread = self.reward_max - self.reward_min
        new_reward = ((unscaled_reward - self.reward_min) / 
                       (spread + tools.EPSILON))
        self.reward_min += spread * self.FORGETTING_RATE
        self.reward_max -= spread * self.FORGETTING_RATE

        # Use change in reward, rather than absolute reward
        delta_reward = new_reward - self.old_reward
        self.old_reward = new_reward
        # Update the reward trace, a brief history of reward
        self.reward_trace.append(delta_reward)
        self.reward_trace.pop(0)
        
        # Gather the cable activities from all the blocks
        cable_index = 0
        block_index = 0
        for block in blocks:
            block_size =  block.cable_activities.size
            self.cable_activities[cable_index: cable_index + block_size] = \
                    block.cable_activities.copy()
            cable_index += block_size 
            block_index += 1

        # Update the reward model.
        # It has a structure similar to the chain transtion model 
        # in daisychain.
        # pre is composed of all the cable activities.
        # post is the selected goal that followed.
        self.chain_activities = self.pre[0] * self.post[0].T
        # Update the count of how often each feature has been active
        self.count = self.count + self.chain_activities
        # Decay the count gradually to encourage occasional re-exploration 
        self.count *= 1. - self.FORGETTING_RATE
        self.count = np.maximum(self.count, 0)
        # Calculate the rate at which to update the reward estimate
        update_rate_raw = (self.chain_activities * ((1 - self.UPDATE_RATE) / 
                                               (self.count + tools.EPSILON) + 
		                                       self.UPDATE_RATE)) 
        update_rate = np.minimum(0.5, update_rate_raw)
        # Collapse the reward history into a single value for this time step
        reward_array = np.array(self.reward_trace)
        # TODO: substitute np.arange in this statement
        decay_exponents = (1. - self.REWARD_DECAY_RATE) ** (
                np.cumsum(np.ones(self.TRACE_LENGTH)) - 1.)
        decayed_array = reward_array.ravel() * decay_exponents
        reward = np.sum(decayed_array.ravel())
        reward_difference = reward - self.expected_reward 
        self.expected_reward += reward_difference * update_rate
        # Decay the reward value gradually to encourage re-exploration 
        self.expected_reward *= 1. - self.FORGETTING_RATE
        # Use the count to estimate the uncertainty in the expected 
        # value of the reward estimate.
        # Use this to scale additive random noise to the reward estimate,
        # encouraging exploration.
        reward_uncertainty = (np.random.normal(size=self.count.shape) *
                              self.EXPLORATION / (self.count + 1.))
        self.estimated_reward_value = self.expected_reward + reward_uncertainty

        # Select a goal cable.
        # First find the estimated reward associated with each chain.   
        chain_votes = (self.cable_activities * self.estimated_reward_value + 
                       tools.EPSILON)
        # Find the maximum estimated reward associated with each potential goal
        hi_end = np.max(chain_votes, axis=0)
        # And the minimum estimated reward associated with each potential goal
        lo_end = np.min(chain_votes, axis=0)
        # Sum the maxes and mins to find the goal with the highest mid-range  
        goal_votes = hi_end + lo_end
        potential_winners = np.where(goal_votes == np.max(goal_votes))[0] 
        # Break any ties by lottery
        winner = potential_winners[np.random.randint(potential_winners.size)]
        # Figure out which block the goal cable belongs to 
        goal_cable = np.remainder(winner, self.cable_activities.size)
        cable_index = goal_cable
        for block in blocks:
            block_size =  block.hub_cable_goals.size
            if cable_index >= block_size:
                cable_index -= block_size
                continue
            else:
                # Activate the goal
                block.hub_cable_goals[cable_index] = 1.
                new_post  = np.zeros(self.post[0].shape)
                new_post[goal_cable] = 1.
                # Remove deliberate goals and actions from pre
                new_pre = np.maximum(self.cable_activities.copy() - 
                                     self.post[-1].copy(), 0.)
                # Update pre and post
                self.pre.append(new_pre)
                self.pre.pop(0)
                self.post.append(new_post)
                self.post.pop(0)
                self._display()
                return
        print 'No goal chosen'
        return 
        
    def add_cables(self, num_new_cables):
        """ Add new cables to the hub when new blocks are created """ 
        self.num_cables = self.num_cables + num_new_cables
        self.expected_reward = tools.pad(self.expected_reward, 
                                      (self.num_cables, self.num_cables), 
                                      val=self.INITIAL_REWARD)
        self.cable_activities = tools.pad(self.cable_activities, 
                                          (self.num_cables, 1))

        self.count = tools.pad(self.count, (self.num_cables, self.num_cables))
        # All the cable activities from all the blocks, at the current time
        for index in range(len(self.pre)):
            self.pre[index] = tools.pad(self.pre[index], (self.num_cables, 1))
            self.post[index] = tools.pad(self.post[index], (self.num_cables, 1))

    def _display(self):
        """ Give a visual update of the internal workings of the hub """
        DISPLAY_PERIOD = 1000.
        #if np.random.random_sample() < 1. / DISPLAY_PERIOD:
        if False:

            # Plot reward value
            fig311 = plt.figure(311)
            plt.gray()
            plt.imshow(self.expected_reward, interpolation='nearest')
            plt.title('reward')
            fig311.show()
            fig311.canvas.draw()
            plt.savefig('log/reward_image.png', bbox_inches=0.)
            
            # Plot weighted chain votes
            fig313 = plt.figure(313)
            plt.gray()
            plt.imshow(np.maximum(self.cable_activities * 
                                  self.estimated_reward_value, 0.), 
                       interpolation='nearest')
            plt.title('cable activities * reward')
            fig313.show()
            fig313.canvas.draw()
            
            # Plot the count 
            fig314 = plt.figure(314)
            plt.gray()
            plt.imshow(1. / (self.count + 1.), interpolation='nearest')
            plt.title('1 / count')
            fig314.show()
            fig314.canvas.draw()
            
            # Plot the reward value plus exploration
            fig315 = plt.figure(315)
            plt.gray()
            plt.imshow(np.maximum(self.estimated_reward_value, 0.), 
                       interpolation='nearest')
            plt.title('estimated_reward_value')
            fig315.show()
            fig315.canvas.draw()
            plt.savefig('log/estimated_reward_image.png', bbox_inches=0.)
            
            # Plot the chain activities 
            fig316 = plt.figure(316)
            plt.gray()
            plt.imshow(self.chain_activities, interpolation='nearest')
            plt.title('chain_activities')
            fig316.show()
            fig316.canvas.draw()

