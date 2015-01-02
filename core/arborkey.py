""" the Arborkey class """
import matplotlib.pyplot as plt
import numpy as np
import tools

class Arborkey(object):
    """ Compares potential goals and decides which to send to the drivetrain
    
    The arborkey is at the highest level of control in the agent.
    It is named for the toothed key used to clamp drill chuck jaws
    around a bit. The arbor key determines which tool is used
    and in which orientation. It is also an intentional reference to the 
    key used to wind a clockwork mechanism,
    due to the fact that it is indispensible and ultimately
    controls whether the mechanism does anything useful.
    
    In the course of each timestep the arborkey
    decides whether to 
    1) pass on the hub's newest goal
    2) pass on a previous hub goal or
    3) don't pass any new goals 
    to the drivetrain.
    
    As it matures, it will also modulate its level of arousal--
    how long it allows itself to evaluate options before taking action--
    based on its recent reward and punishment history.
    """
    def __init__(self):
        self.MAX_LENGTH = 25
        self.REWARD_DECAY_RATE = .5
        self.ACTION_PROPENSITY = 4. / float(self.MAX_LENGTH)
        self.goal_candidates = []
        self.expected_reward = []
        self.time_since_observed = []
        self.time_since_acted = 0.

    def step(self, goal_candidate, candidate_reward, current_reward):
        """ Evaluate and recommend a goal candidate """
        self.time_since_acted += 1.
        goal = None
        # Update the list of goal candidates 
        self.goal_candidates.append(goal_candidate)
        self.expected_reward.append(candidate_reward)
        self.time_since_observed.append(0.)
        # Estimate the value of each candidate
        decayed_reward = np.array(self.expected_reward) / ( 1. + 
                self.REWARD_DECAY_RATE * np.array(self.time_since_observed))
        value = (decayed_reward - current_reward + 
                 self.ACTION_PROPENSITY * self.time_since_acted)
        # Find the most likely candidate
        best_goal_index = np.where(value == max(value))[0][-1]
        highest_value = value[best_goal_index]
        # Check whether the best candidate is good enough to pick 
        if highest_value > 0.:
            goal = self.goal_candidates.pop(best_goal_index)
            self.expected_reward.pop(best_goal_index)
            self.time_since_observed.pop(best_goal_index)
            self.goal_candidates = []
            self.expected_reward = []
            self.time_since_observed = []
            self.time_since_acted = 0.
        # If the list of candidates is too long, reduce it
        if len(self.goal_candidates) > self.MAX_LENGTH:
            worst_goal_index = np.where(self.expected_reward == 
                                        min(self.expected_reward))[0][0]
            self.goal_candidates.pop(worst_goal_index)
            self.expected_reward.pop(worst_goal_index)
            self.time_since_observed.pop(worst_goal_index)
        return goal

    def add_cables(self, num_new_cables):
        """ Add new cables to the hub when new gearboxes are created """ 
        pass

    def visualize(self):
        pass
