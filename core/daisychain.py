import tools

import numpy as np

class DaisyChain(object):
    """
    An incremental model-based reinforcement learning algorithm
    
    Input channels are chained together in two-element sequences,
    and an expected reward is associated with each.
    A helpful metaphor is joining cables end-to-end in a daisy chain,
    each containing a pre cable and a post cable.
    If activity in the post cable follows activity in pre cable, 
    the activity in the chain they share is high.

    The daisychain is the basis for prediction in the agent. 
    It incrementally estimates the likelihood that a post cable 
    will be active given the set of pre cables that have recently
    been active. 

    Each pre-post chain also tracks the expected reward, the uncertainty in
    the estimates of the post activity and reward, and a count of how many
    times the chain has been active.
    """
    def __init__(self, max_num_cables, name):
        """ Initialize the daisychain, preallocating all data structures """
        self.max_num_cables = max_num_cables
        self.name = name

        # User-defined constants
        self.AGING_TIME_CONSTANT = 10 ** 6 # real, large
        # debug was -1
        self.CHAIN_UPDATE_RATE = 10 ** -5 # real, 0 < x < 1
        self.VOTE_DECAY_RATE = 0.1 # real, 0 < x < 1
        self.INITIAL_UNCERTAINTY = 0.5 # real, 0 < x < 1
        
        self.time_steps = 0
        daisychain_shape = (max_num_cables, max_num_cables)        
        self.count = np.zeros(daisychain_shape)
        self.pre_count = np.zeros(daisychain_shape)
        self.expected_post = np.zeros(daisychain_shape)
        self.post_uncertainty = np.zeros(daisychain_shape)
        self.reward_value = np.zeros(daisychain_shape)
        self.reward_uncertainty = (np.ones(daisychain_shape) *
				  self.INITIAL_UNCERTAINTY)
        state_shape = (max_num_cables,1)
        self.pre = np.zeros(state_shape)
        self.post = np.zeros(state_shape)
        self.num_cables = 0
             
        self.deliberation_vote = np.zeros((max_num_cables, 1))
        self.surprise = np.ones((max_num_cables, 1))

    def update(self, cable_activities, reward):        
        """ Train the daisychain using the current cable_activities 
        and reward """
        self.num_cables = np.maximum(self.num_cables, cable_activities.size)
        # Pad the incoming cable_activities array out to its full size 
        cable_activities = tools.pad(cable_activities, 
                                     (self.max_num_cables, 0))
        self.current_reward = reward
        # The pre is a weighted sum of previous cable_activities, with the most
        # recent cable_activities being weighted the highest
        # debug
        #self.pre = tools.bounded_sum([
        #        self.post, self.pre * (1 - self.PRE_DECAY_RATE)])
        self.pre = self.post
        self.post = cable_activities
        chain_activities = self.pre * self.post.T
        chain_activities[np.nonzero(np.eye(self.pre.size))] = 0.
        update_rate_raw = (chain_activities * 
                           ((1 - self.CHAIN_UPDATE_RATE) / 
                            (self.count + tools.EPSILON) + 
                            self.CHAIN_UPDATE_RATE))
        update_rate = np.minimum(0.5, update_rate_raw)
        self.count += chain_activities
        self.count -= 1 / (self.AGING_TIME_CONSTANT * self.count + 
                           tools.EPSILON)
        self.count = np.maximum(self.count, 0)
        reward_difference = np.abs(reward - self.reward_value)
        self.reward_value += (reward - self.reward_value) * update_rate
        self.reward_uncertainty += (reward_difference - 
                                    self.reward_uncertainty) * update_rate
        update_rate_raw_post = (self.pre * ((1 - self.CHAIN_UPDATE_RATE) / 
                                            (self.pre_count + tools.EPSILON) + 
		                                    self.CHAIN_UPDATE_RATE)) 
        update_rate_post = np.minimum(0.5, update_rate_raw_post)
        self.pre_count += self.pre
        self.pre_count -= 1 / (self.AGING_TIME_CONSTANT * self.pre_count +
                               tools.EPSILON)
        self.pre_count = np.maximum(self.pre_count, 0)
        post_difference = np.abs(self.pre * self.post.T - self.expected_post)
        self.expected_post += (self.pre * self.post.T - 
		               self.expected_post) * update_rate_post
        self.post_uncertainty += (post_difference - 
                                  self.post_uncertainty) * update_rate_post 
        # Reaction is the expected post, turned into a deliberation_vote
        self.reaction = tools.weighted_average(self.expected_post, self.pre)
        # Surprise is the difference between the expected post and
        # the actual one
        self.surprise = tools.weighted_average(
                np.abs(self.post.T - self.expected_post), 
		        self.pre / (self.post_uncertainty + tools.EPSILON))
        #self.surprise = tools.weighted_average(
        #        np.abs((self.post.T - self.expected_post) / 
		#               (self.post_uncertainty + tools.EPSILON)), 
		#        self.pre / (self.post_uncertainty + tools.EPSILON))
        # Reshape chain activities into a single column
        return chain_activities.ravel()[:,np.newaxis]
   
    def deliberate(self, goal_value_by_chain):
        """ Choose goals deliberatively, based on deliberation_vote i
        and reward value """
        # Maintain the internal deliberation_vote set
        deliberation_vote_fulfillment = 1 - self.post
        deliberation_vote_decay = 1 - self.VOTE_DECAY_RATE
        self.deliberation_vote *= (deliberation_vote_fulfillment * 
                                   deliberation_vote_decay)

        similarity = np.tile(self.post, (1,self.post.size))
        reward_noise = (np.random.random_sample(
                self.reward_uncertainty.shape)* 2 - 1)
        estimated_reward_value = (self.reward_value - self.current_reward + 
                                  self.reward_uncertainty * reward_noise)
        estimated_reward_value = np.maximum(estimated_reward_value, 0)
        estimated_reward_value = np.minimum(estimated_reward_value, 1)
        reward_value_by_cable = tools.weighted_average(
                estimated_reward_value, 
                similarity / (self.reward_uncertainty + tools.EPSILON))
        reward_value_by_cable[self.num_cables:] = 0. 
        # Reshape goal_value_by_chain back into a square array 
        goal_value_by_chain = np.reshape(goal_value_by_chain, 
                                         (self.deliberation_vote.size, -1))
        # Bounded sum of the deliberation_vote values from above over all chains 
        goal_value_by_cable = tools.bounded_sum(goal_value_by_chain.T * 
                                             similarity)
        count_by_cable = tools.weighted_average(self.count, similarity)
        exploration_vote = ((1 - self.current_reward) / 
                (self.num_cables * (count_by_cable + 1) * 
                 np.random.random_sample(count_by_cable.shape) + tools.EPSILON))
        exploration_vote = np.minimum(exploration_vote, 1.)
        exploration_vote[self.num_cables:] = 0.
        #exploration_vote = np.zeros(reward_value_by_cable.shape)
        # debug
        include_goals = True
        if include_goals:
            #total_vote = (reward_value_by_cable + goal_value_by_cable +
            #              exploration_vote)
            cable_goals = tools.bounded_sum([reward_value_by_cable, 
                                   goal_value_by_cable, exploration_vote])
        else:
            #total_vote = reward_value_by_cable + exploration_vote
            cable_goals = tools.bounded_sum([reward_value_by_cable, exploration_vote])
        self.deliberation_vote = np.maximum(cable_goals, self.deliberation_vote)
        # TODO perform deliberation centrally at the guru and 
        # modify cable goals accordingly. In this case cable_goals
        # will be all reactive, except for the deliberative component
        # from the guru.
        return cable_goals[:self.num_cables]

    def get_cable_deliberation_vote(self):
        return self.deliberation_vote[:self.num_cables]

    def get_cable_activity_reactions(self):
        return self.reaction[:self.num_cables]

    def get_surprise(self):
        return self.surprise[:self.num_cables]

    def get_projection(self, map_projection):
        """ Find the projection from chain activities to cable signals """
        num_cables = self.reward_value.shape[0]
        projection = np.zeros((num_cables,2))
        chains = np.reshape(map_projection, (num_cables,num_cables))
        projection[:,0] = np.sign(np.max(chains, axis=1))
        projection[:,1] = np.sign(np.max(chains, axis=0))
        return projection
    
    def visualize(self, save_eps=True):
        """ Show the internal state of the daisychain in a pictorial format """
        tools.visualize_array(self.reward_value, 
                                  label=self.name + '_reward')
        #tools.visualize_array(self.reward_uncertainty, 
        #                          label=self.name + '_reward_uncertainty')
        tools.visualize_array(np.log(self.count + 1.), 
                                  label=self.name + '_count')
        #tools.visualize_daisychain(self, self.num_primitives, 
        #                          self.num_actions, 10)
        return
