""" the Mainspring class """
import copy
import matplotlib.pyplot as plt
import numpy as np
import tools

class Mainspring(object):
    """
    long term abstract memory and evaluating potential goals
    
    The mainspring is named after the central energy storage mechanism
    in a clockwork machine, due to its important role in maintaining
    short term memory, long term memory and planning.

    In each timestep, the mainspring takes several actions:
    * It takes in the current attended cable.
    * It updates short term memory, a record of recently observed cables.
    * It updates long term memory, a record of observed cable transitions.
    * It updates its estimate of the reward associated with each transition.

    The mainspring can also be queried to provide:
    * an estimate of reward, given a goal candidate (from the arborkey),
    * an prediction of future cables (from the spindle)
    """
    def __init__(self, initial_size):
        self.UPDATE_RATE = 1e-2
        self.REWARD_LEARNING_RATE = 1e-1
        self.INITIAL_REWARD = .1
        # Keep a history of reward and active cables to account for 
        # delayed reward.
        self.TRACE_LENGTH = 11
        # How many recently attended cables to hold in short term memory
        self.NUM_ATTENDED = 25
        self.num_cables = initial_size
        cable_shape = (self.num_cables, 1)
        transition_shape = (self.num_cables, self.num_cables)

        self.reward_history = list(np.zeros(self.TRACE_LENGTH))
        cable_shape = (self.num_cables, 1)
        self.stm_indices = [[0.] * 
                self.NUM_ATTENDED] * self.TRACE_LENGTH
        self.stm_activities = [[0.] * 
                self.NUM_ATTENDED] * self.TRACE_LENGTH
        self.time_since_attended = [[1.] * 
                self.NUM_ATTENDED] * self.TRACE_LENGTH
        self.value = np.zeros(transition_shape)
        self.reward = np.ones(transition_shape) * self.INITIAL_REWARD

    def step(self, new_index, new_activity, raw_reward):
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

        # Update long term memory based on the newest attended cable and
        # short term memory from self.TRACE_LENGTH time steps ago.
        # For each element in short term memory, update its relationship
        # with the most recently attended cable   
        # Calculate the decayed activity of the cables
        decayed_activities = self.get_decayed_activities(0)
        trace_indices = self.stm_indices[0]
        for i in np.arange(self.NUM_ATTENDED):
            # Find the difference between the expected and observed values
            delta_activity = (decayed_activities[i] - 
                              self.value[trace_indices[i], new_index])
            # Then step toward that value by a small amount
            self.value[trace_indices[i], new_index] += (delta_activity * 
                    self.UPDATE_RATE * decayed_activities[i] ** 2)
            # Update the reward estimate
            # Find the difference between the estimated and observed reward
            delta_reward = reward_trace - self.reward[trace_indices[i], 
                                                      new_index]
            self.reward[trace_indices[i], new_index] += (delta_reward * 
                    self.REWARD_LEARNING_RATE * decayed_activities[i] ** 2)
        # Update the traces. Remove the oldest element and create a new one.
        self.stm_indices.pop(0)
        self.stm_activities.pop(0)
        self.time_since_attended.pop(0)
        self.stm_indices.append(copy.deepcopy(self.stm_indices[-1]))
        self.stm_activities.append(copy.deepcopy(self.stm_activities[-1]))
        self.time_since_attended.append(copy.deepcopy(
                self.time_since_attended[-1]))
        # Update short term memory in the most recent trace element
        # Find an attended cable to remove
        drop_index = np.argmin(decayed_activities)
        self.stm_indices[-1].pop(drop_index)
        self.stm_activities[-1].pop(drop_index)
        self.time_since_attended[-1].pop(drop_index)
        # Add the newly attended cable 
        self.stm_indices[-1].append(new_index)
        self.stm_activities[-1].append(new_activity)
        self.time_since_attended[-1].append(0.)
        for cable in range(len(self.time_since_attended[-1])):
            self.time_since_attended[-1][cable] += 1.
        #self.visualize()

    def evaluate(self, goal):
        """ Given the index of a goal, calculate the expected reward """
        # Pick out the most current set of attended cables
        recently_attended = self.stm_indices[-1]
        # Calculate the most recent set of decayed cable activities
        decayed_activities = self.get_decayed_activities(-1)
        # Pick out the reward values that are relevant
        relevant_rewards = self.reward[recently_attended, goal]
        # Weight them according to the cables' decayed activities
        weighted_reward = decayed_activities * relevant_rewards
        expected_reward = (np.sum(weighted_reward, axis=0) / 
                           (np.sum(decayed_activities) + tools.EPSILON))
        return expected_reward

    def predict(self):
        """ Given the current set of recent cables, predict future ones """
        # TODO 
        return

    def get_decayed_activities(self, index=-1):
        """ Calculate the decayed activites of the recently attended cables.
        There are a number of viable decay functions, but it can be 
        useful to have one with a very long tail. To get these,
        the 1/t decay is used.
        """
        # Calculate decayed activities
        recency_factor = 1. / (np.array(self.time_since_attended[index]) +
                               tools.EPSILON)
        decayed_activities = (np.array(self.stm_activities[index]) * 
                              recency_factor)
        return decayed_activities

    def add_cables(self, num_new_cables):
        """ Add new cables to the hub when new gearboxes are created """ 
        self.num_cables = self.num_cables + num_new_cables
        transition_shape = (self.num_cables, self.num_cables)
        self.value = tools.pad(self.value, transition_shape)
        self.reward = tools.pad(self.reward, transition_shape)

    def visualize(self):
        # Plot reward value
        plt.figure(311)
        plt.subplot(1,2,1)
        plt.gray()
        plt.imshow(self.value.astype(np.float), interpolation='nearest')
        plt.title('mainspring value')
        plt.show()
        #plt.savefig('log/reward_image.png', bbox_inches=0.)
