import numpy as np

import tools

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
        self.CHAIN_UPDATE_RATE = 10 ** -1 # real, 0 < x < 1
        # Initialize variables
        self.time_steps = 0
        daisychain_shape = (max_num_cables, max_num_cables)        
        self.count = np.zeros(daisychain_shape)
        self.expected_cable_activities = np.zeros(daisychain_shape)
        self.post_uncertainty = np.zeros(daisychain_shape)
        state_shape = (max_num_cables,1)
        self.pre = np.zeros(state_shape)
        self.pre_count = np.zeros(state_shape)
        self.post = np.zeros(state_shape)
        self.num_cables = 0
        #self.deliberation_vote = np.zeros((max_num_cables, 1))
        self.surprise = np.ones((max_num_cables, 1))

    def step_up(self, cable_activities):        
        """ Train the daisychain using the current cable_activities """
        self.num_cables = np.maximum(self.num_cables, cable_activities.size)
        # Pad the incoming cable_activities array out to its full size 
        cable_activities = tools.pad(cable_activities, 
                                     (self.max_num_cables, 0))
        self.pre = self.post.copy()
        self.post = cable_activities.copy()
        chain_activities = self.pre * self.post.T
        chain_activities[np.nonzero(np.eye(self.pre.size))] = 0.
        self.count += chain_activities
        self.count -= 1 / (self.AGING_TIME_CONSTANT * self.count + 
                           tools.EPSILON)
        self.count = np.maximum(self.count, 0)
        update_rate_raw_post = (self.pre * ((1 - self.CHAIN_UPDATE_RATE) / 
                                            (self.pre_count + tools.EPSILON) + 
		                                    self.CHAIN_UPDATE_RATE)) 
        update_rate_post = np.minimum(0.5, update_rate_raw_post)
        self.pre_count += self.pre
        self.pre_count -= 1 / (self.AGING_TIME_CONSTANT * self.pre_count +
                               tools.EPSILON)
        self.pre_count = np.maximum(self.pre_count, 0)
        post_difference = np.abs(self.pre * self.post.T - 
                                 self.expected_cable_activities)
        self.expected_cable_activities += update_rate_post * (
                self.pre * self.post.T - self.expected_cable_activities)
        self.post_uncertainty += (post_difference - 
                                  self.post_uncertainty) * update_rate_post 
        # Reaction is the expected post, turned into a deliberation_vote
        self.reaction = tools.weighted_average(self.expected_cable_activities, 
                                               self.pre)
        # Surprise is the difference between the expected post and
        # the actual one
        self.surprise = tools.weighted_average(
                np.abs(self.post.T - self.expected_cable_activities), 
		        self.pre / (self.post_uncertainty + tools.EPSILON))
        # Reshape chain activities into a single column
        return chain_activities.ravel()[:,np.newaxis]
   
    def step_down(self, chain_goals):
        """ Propogate goals down through the transition model """
        # Reshape chain_goals back into a square array 
        chain_goals = np.reshape(chain_goals, (self.post.size, -1))
        # Weight chain goals by the current cable activities   
        upstream_goals = tools.bounded_sum(self.post * chain_goals.T)
        cable_goals = tools.bounded_sum([upstream_goals, self.reaction])
        return cable_goals[:self.num_cables]

    def get_surprise(self):
        return self.surprise[:self.num_cables]

    def get_index_projection(self, map_projection):
        """ Find the projection from chain activities to cable signals """
        num_cables = np.int(map_projection.size ** .5)
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
