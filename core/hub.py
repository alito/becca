import matplotlib.pyplot as plt
import numpy as np

import tools

class Hub(object):
    """ The hub is the central action selection mechanism 
    
    The analogy of the hub and spoke stucture is suggested by 
    the fact that the hub has a separate connection to each
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
        self.INITIAL_VARIANCE = 2. # real, 0 < x < 1
        self.INITIAL_REWARD = 1.0 # real, 0 < x < 1
        self.UPDATE_RATE = 10 ** -1.
        self.REWARD_DECAY_RATE = .5 # real, 0 < x < 1
        self.TRACE_LENGTH = 5
        self.EXPLORATION = 0.
        #self.CONSISTENCY_WEIGHT = 10. 
        self.VARIANCE_PENALTY_CONSTANT = .1
        self.VARIANCE_PENALTY_STOCHASTIC = 1.
        #self.UNC_WEIGHT = 1./3.
        #self.UNC_WEIGHT = 1./10.

        self.REWARD_RANGE_DECAY_RATE = 10 ** -5
        self.reward_min = tools.BIG
        self.reward_max = -tools.BIG
        self.old_reward = 0.
        self.count = np.zeros((self.num_cables, self.num_cables))
        self.reward_trace = [0.] * self.TRACE_LENGTH
        self.reward_value = (np.ones((self.num_cables, self.num_cables)) *
                             self.INITIAL_REWARD)
        self.reward_variance = (np.ones((self.num_cables, self.num_cables)) *
				                self.INITIAL_VARIANCE)
        # All the cable activities from all the blocks, at the current time
        self.pre = [np.zeros((self.num_cables, 1))] * self.TRACE_LENGTH
        self.post = [np.zeros((self.num_cables, 1))] * self.TRACE_LENGTH
        self.cable_activities = np.zeros((self.num_cables, 1))
        return
    
    def step(self, blocks, unscaled_reward):
        """ Advance the hub one step:
        1. Comb tower of blocks, collecting cable activities from each
        2. Update all-to-all reward model
        3. Select a goal to up- (or down-) vote
        4. Modify the goal in the block
        """
        # Adapt the reward so that it falls between -1 and 1 
        self.reward_min = np.minimum(unscaled_reward, self.reward_min)
        self.reward_max = np.maximum(unscaled_reward, self.reward_max)
        spread = self.reward_max - self.reward_min
        new_reward = ((unscaled_reward - self.reward_min) / 
                       (spread + tools.EPSILON))
        self.reward_min += spread * self.REWARD_RANGE_DECAY_RATE
        self.reward_max -= spread * self.REWARD_RANGE_DECAY_RATE

        # Using change in reward really matters
        delta_reward = new_reward - self.old_reward
        self.old_reward = new_reward
        self.reward_trace.pop(0)
        self.reward_trace.append(delta_reward)
        
        #self.reward_trace.append(new_reward)
        # pre is composed of all the cable activities
        # post is the selected goal that followed
        self.pre.pop(0)
        self.pre.append(self.cable_activities.copy())
        # Collect all the cable activities
        cable_index = 0
        for block in blocks:
            block_size =  block.cable_activities.size
            self.cable_activities[cable_index: cable_index + block_size] = \
                    block.cable_activities.copy()
            cable_index += block_size 

        # Update the reward model.
        # It has a structure similar to the chain transtion model in daisychain
        chain_activities = self.pre[0] * self.post[0].T
        #print 'cha', np.nonzero(chain_activities)[0]
        #print 'cha', chain_activities[:9,:9]
        self.count = self.count + chain_activities
        #print 'cnt', self.count[np.nonzero(self.count)]
        self.count = np.maximum(self.count, 0)
        update_rate_raw = (chain_activities * ((1 - self.UPDATE_RATE) / 
                                               (self.count + tools.EPSILON) + 
		                                       self.UPDATE_RATE)) 
        update_rate = np.minimum(0.5, update_rate_raw)
        #print 'ur', update_rate
        reward_array = np.array(self.reward_trace)
        decay_exponents = (1. - self.REWARD_DECAY_RATE) ** (
                np.cumsum(np.ones(self.TRACE_LENGTH)) - 1.)
        decayed_array = reward_array.ravel() * decay_exponents
        reward = tools.bounded_sum(decayed_array.ravel())
        #if reward:
        #    print 'reward =', reward
        #    print 'pre', self.pre[0].ravel()
        #    print 'post', self.post[0].ravel()
        reward_difference = reward - self.reward_value 
        #i = np.where(np.abs(reward_difference) > 1.)
        #if i[0].size > 0:
        #    print '---'
        #    print 'rd', reward_difference[i]
        #    print 'r', reward
        #    print 'rv', self.reward_value[i]
        #    print 'ur', update_rate[i]
        self.reward_value += reward_difference * update_rate
        #fig411 = plt.figure(411)
        #plt.gray()
        #plt.imshow(self.reward_value, interpolation='nearest')
        #plt.title('reward')
        #fig411.show()
        #fig411.canvas.draw()
        #print 'rv', self.reward_value[:18,:18]
        #print 'reward', reward, 'reward_trace', self.reward_trace
        #print 'pre', self.pre[0].ravel(), 'post', self.post[0].ravel()
        show = False
        if np.random.random_sample() < 0.001:
            show = True

        self.reward_variance += (np.abs(reward_difference) - 
                                    self.reward_variance) * update_rate
        #print 'varmax', np.max(self.reward_variance)
        #reward_uncertainty = (np.random.normal(size=self.count.shape) ** 2. *
        #                      self.UNC_WEIGHT / (self.count + 1.))
        #reward_uncertainty = (np.random.normal(size=self.count.shape) *
        #                      (1. / self.count.shape[1] ** .2) *
        #                      self.EXPLORATION / (self.count + 1.))
        #reward_uncertainty = (np.random.normal(size=self.count.shape) *
        #                      self.EXPLORATION / (self.count + 1.))
        reward_uncertainty = (np.random.normal(size=self.count.shape) *
                              self.EXPLORATION / (self.count + 1.))

        #variance_penalty = self.reward_variance * np.random.normal(
        #        scale=self.EXPLORATION, size=self.reward_variance.shape) ** 2.
        # TODO: Add constants
        #consistency_reward = self.CONSISTENCY_WEIGHT / ( 
        #        10. * self.reward_variance + 1.)
        #consistency_reward = self.CONSISTENCY_WEIGHT * self.reward_variance
        variance_penalty = self.reward_variance * (
                self.VARIANCE_PENALTY_CONSTANT + 
                self.VARIANCE_PENALTY_STOCHASTIC * 
                np.abs(np.random.normal(size=self.reward_variance.shape)))
        #print 'rn', variance_penalty
        #estimated_reward_value = (self.reward_value + consistency_reward + 
        #                          reward_uncertainty)
        estimated_reward_value = (self.reward_value - variance_penalty)
        #estimated_reward_value = (self.reward_value - consistency_reward + 
        #                          reward_uncertainty)
        #estimated_reward_value = (self.reward_value - variance_penalty + 
        #                          reward_uncertainty)
        #print 'erv', estimated_reward_value
        # Select a goal cable
        flattened_cable_votes = (self.cable_activities * 
                                 estimated_reward_value + 
                                 tools.EPSILON).ravel()
        #winner = np.argmax(flattened_cable_votes) 
        potential_winners = np.where(flattened_cable_votes == 
                                     np.max(flattened_cable_votes))[0] 
        winner = potential_winners[np.random.randint(potential_winners.size)]
        goal_cable = np.remainder(winner, self.cable_activities.size)
        #print 'winner', winner,'goal_cable', goal_cable,  'row', winner/ self.cable_activities.size
        #print 'reward', self.reward_value.ravel()[winner], 'unc', self.reward_variance.ravel()[winner], 'vote', flattened_cable_votes[winner]
        #print 'max_vote', np.max(flattened_cable_votes), 'max_unc', np.max(self.reward_variance), 'max rew', np.max(self.reward_value)
        if show:
            fig311 = plt.figure(311)
            plt.gray()
            plt.imshow(self.reward_value, interpolation='nearest')
            plt.title('reward')
            fig311.show()
            fig311.canvas.draw()
            plt.savefig('log/reward_image.png', bbox_inches=0.)
            fig312 = plt.figure(312)
            plt.gray()
            plt.imshow(self.reward_variance, interpolation='nearest')
            plt.title('reward uncertainty')
            fig312.show()
            fig312.canvas.draw()
            plt.savefig('log/reward_uncertainty_image.png', bbox_inches=0.)
            fig313 = plt.figure(313)
            plt.gray()
            plt.imshow(np.maximum(self.cable_activities * 
                                  estimated_reward_value, 0.), 
                       interpolation='nearest')
            plt.title('cable activities * reward')
            fig313.show()
            fig313.canvas.draw()
            fig314 = plt.figure(314)
            plt.gray()
            plt.imshow(1. / (self.count + 1.), interpolation='nearest')
            plt.title('1 / count')
            fig314.show()
            fig314.canvas.draw()
            fig315 = plt.figure(315)
            plt.gray()
            plt.imshow(np.maximum(estimated_reward_value, 0.), interpolation='nearest')
            plt.title('estimated_reward_value')
            fig315.show()
            fig315.canvas.draw()
            plt.savefig('log/estimated_reward_image.png', bbox_inches=0.)
            fig316 = plt.figure(316)
            plt.gray()
            plt.imshow(chain_activities, interpolation='nearest')
            plt.title('chain_activities')
            fig316.show()
            fig316.canvas.draw()

        cable_index = goal_cable
        for block in blocks:
            block_size =  block.hub_cable_goals.size
            if cable_index >= block_size:
                cable_index -= block_size
                continue
            else:
                block.hub_cable_goals[cable_index] = 1.
                new_post  = np.zeros(self.post[0].shape)
                new_post[goal_cable] = 1.
                self.post.pop(0)
                self.post.append(new_post)
                #print 'cable', goal_cable, 'chosen in', block.name
                return
        print 'No goal chosen'
        return 
        
    def add_cables(self, num_new_cables):
        """ Add new cables to the hub when new blocks are created """ 
        self.num_cables = self.num_cables + num_new_cables
        self.reward_value = tools.pad(self.reward_value, 
                                      (self.num_cables, self.num_cables), 
                                      val=self.INITIAL_REWARD)
        self.reward_variance = tools.pad(self.reward_variance, 
                                            (self.num_cables, self.num_cables),
                                            val=self.INITIAL_VARIANCE)
        self.cable_activities = tools.pad(self.cable_activities, 
                                          (self.num_cables, 1))

        self.count = tools.pad(self.count, (self.num_cables, self.num_cables))
        # All the cable activities from all the blocks, at the current time
        for index in range(len(self.pre)):
            self.pre[index] = tools.pad(self.pre[index], (self.num_cables, 1))
            self.post[index] = tools.pad(self.post[index], (self.num_cables, 1))
