import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np

from cog import Cog
from level import Level
import utils as ut

class Agent(object):
    """ 
    A general reinforcement learning agent
    
    Takes in a time series of sensory input vectors and 
    a scalar reward and puts out a time series of action commands.
    """
    def __init__(self, num_sensors, num_actions, show=True, agent_name='my'):
        """
        Configure the Agent

        num_sensors and num_actions are the only absoluetly necessary
        arguments. They define the number of elements in the 
        sensors and actions arrays that the agent and the world use to
        communicate with each other. 
        """
        self.VISUALIZE_PERIOD = 10 ** 3
        self.BACKUP_PERIOD = 10 ** 4
        self.show = show
        self.pickle_filename ="log/" + agent_name + "_agent.pickle"
        # TODO: Automatically adapt to the number of sensors pass in
        self.num_sensors = num_sensors
        self.num_actions = num_actions

        # Initialize agent infrastructure
        self.num_levels =  1
        self.levels = [Level(name='level'+str(self.num_levels - 1))]
        self.action = np.zeros((self.num_actions,1))
        # Initialize constants for adaptive reward scaling 
        self.REWARD_RANGE_DECAY_RATE = 10 ** -5
        self.reward_min = ut.BIG
        self.reward_max = -ut.BIG
        self.reward = 0
        self.cumulative_reward = 0
        self.reward_history = []
        self.reward_steps = []
        self.surprise_history = []
        self.timestep = 0
        self.graphing = True

    def step(self, sensors, unscaled_reward):
        """ Step through one time interval of the agent's operation """
        self.timestep += 1
        if sensors.ndim == 1:
            sensors = sensors[:,np.newaxis]
        # Modify the reward so that it automatically falls between 0 and 1 
        self.reward_min = np.minimum(unscaled_reward, self.reward_min)
        self.reward_max = np.maximum(unscaled_reward, self.reward_max)
        spread = self.reward_max - self.reward_min
        self.reward = ((unscaled_reward - self.reward_min) / 
                       (spread + ut.EPSILON))
        self.reward_min += spread * self.REWARD_RANGE_DECAY_RATE
        self.reward_max -= spread * self.REWARD_RANGE_DECAY_RATE

        # Propogate the new sensor inputs up through the levels
        feature_inputs = np.vstack((self.action, sensors))
        for level in self.levels:
            feature_inputs = level.step_up(feature_inputs, self.reward) 
        # Create a new level if needed
        if feature_inputs.size > 0:
            self.num_levels +=  1
            self.levels.append(Level(name='level'+str(self.num_levels - 1)))
            feature_inputs = self.levels[-1].step_up(feature_inputs, 
                                                     self.reward) 
            print "Added level", self.num_levels - 1

        # Propogate the goals down through the levels
        max_surprise = 0.0
        goals = np.zeros((feature_inputs.size,1))
        for level in reversed(self.levels):
            goal_vote = level.step_down(goals)
            goals = level.goal_output
            if level.surprise.size > 0:
                #if not np.isnan(np.max(level.surprise)): 
                max_surprise = np.maximum(np.max(level.surprise), 
                                          max_surprise)
                max_arg = np.argmax(level.surprise)
        self.surprise_history.append(max_surprise)

        # Strip the actions off the goals to make the current set of actions.
        # For actions, each goal is a probability threshold. If a roll of
        # dice comes up lower than thegoal value, the action is taken
        # with a magnitude of 1.
        if goals.size < self.num_actions:
            goals = ut.pad(goals,(self.num_actions, 0))
        self.action = np.zeros((self.num_actions, 1))
        action_thresholds = np.random.random_sample((self.num_actions, 1))
        self.action[np.nonzero(goals[:self.num_actions,:] 
                    > action_thresholds)] = 1.
        if (self.timestep % self.BACKUP_PERIOD) == 0:
                self._save()    
        self._display(unscaled_reward)
        return self.action

    def get_projections(self, to_screen=False):
        """
        Get representations of all the features in each level 
        
        Every feature is projected down through its own level and
        the levels below it until its projection on sensor inputs 
        and actions is obtained. This is a way to represent the
        receptive field of each feature.

        Returns a list containing the projection for each feature 
        in each level.
        """
        all_projections = []
	for level_index in range(len(self.levels)):
            level_projections = []
            num_features = self.levels[level_index].output_map.shape[0]
            for feature_index in range(num_features):    
                features = np.zeros((num_features, 1))
                features[feature_index, 0] = 1.
                projection = self._get_projection(level_index, features)
                level_projections.append(projection)
                # Display the projection in text form if desired
                if to_screen:
                    print 'projection', self.levels[level_index].name, \
                            'feature', feature_index
                    for i in range(projection.shape[1]):
                        print np.nonzero(projection)[0][np.where(np.nonzero(
                                                         projection)[1] == i)]
            if len(level_projections) > 0:
                all_projections.append(level_projections)
        return all_projections
  
    def _get_projection(self, level_index, feature_values):
        """
        Get the projection for feature_values
        
        Recursively projects feature_values down through levels
        until the bottom level is reached. Feature values is a 
        two-dimensional array and can contain
        several columns. Each column represents a state, and their
        order represents a temporal progression. During projection
        to the next lowest level, the number of states
        increases by one. 
        
        Returns the projection in terms of basic sensor inputs and actions. 
        """
        if level_index == -1:
            return feature_values
        projection = np.zeros((self.levels[level_index].num_feature_inputs, 
                               feature_values.shape[1] + 1))
        for feature_index in range(feature_values.shape[0]):
            for state_index in range(feature_values.shape[1]):
                if feature_values[feature_index, state_index] > 0:
                    new_contribution = self.levels[
                            level_index].get_projection(feature_index)
                    projection[:,state_index:state_index + 2] = np.maximum(
                            projection[:,state_index:state_index + 2], 
                            new_contribution)
        projection = self._get_projection(level_index - 1, projection)
        return projection

    def _display(self, unscaled_reward):
        """ Show the current state and some history of the agent """
        self.cumulative_reward += unscaled_reward
        if (self.timestep % self.VISUALIZE_PERIOD) == 0:
            print self.timestep, 'time steps'
            self.reward_history.append(float(self.cumulative_reward) / 
                                       self.VISUALIZE_PERIOD)
            self.cumulative_reward = 0    
            self.reward_steps.append(self.timestep)
            self._show_reward_history(save_eps=True)
        return
 
    def report_performance(self):
        """ Report on the reward amassed by the agent """
        performance = np.mean(self.reward_history)
        print("Final performance is %f" % performance)
        self._show_reward_history(save_eps=True, block=self.show)
        return performance
    
    def _show_reward_history(self, block=False, save_eps=False,
                            epsfilename='log/reward_history.eps'):
        """ Show the agent's reward history and save it to a file """
        if self.graphing:
            fig = plt.figure(1)
            plt.plot(self.reward_steps, self.reward_history)
            plt.xlabel("time step")
            plt.ylabel("average reward")
            fig.show()
            fig.canvas.draw()
            if save_eps:
                plt.savefig(epsfilename, format='eps')
            if block:
                plt.show()
        return
    
    def _save(self):
        """ Archive a copy of the agent object for future use """
        success = False
        try:
            with open(self.pickle_filename, 'wb') as agent_data:
                pickle.dump(self, agent_data)
            print("Agent data saved at " + str(self.timestep) + " time steps")
        except IOError as err:
            print("File error: " + str(err) + 
                  " encountered while saving agent data")
        except pickle.PickleError as perr: 
            print("Pickling error: " + str(perr) + 
                  " encountered while saving agent data")        
        else:
            success = True
        return success
        
    def restore(self):
        """ Reconstitute the agent from a previously saved agent """
        restored_agent = self
        try:
            with open(self.pickle_filename, 'rb') as agent_data:
                loaded_agent = pickle.load(agent_data)

            # Compare the number of channels in the restored agent with 
            # those in the already initialized agent. If it matches, 
            # accept the agent. If it doesn't,
            # print a message, and keep the just-initialized agent.
            if((loaded_agent.num_sensors == self.num_sensors) and 
               (loaded_agent.num_actions == self.num_actions)):
                print("Agent restored at timestep " + 
                      str(loaded_agent.timestep))
                restored_agent = loaded_agent
            else:
                print("The agent " + self.pickle_filename + " does not have " +
                      "the same number of input and output elements as " + 
                      "the world.")
                print("Creating a new agent from scratch.")
        except IOError:
            print("Couldn't open %s for loading" % self.pickle_filename)
        except pickle.PickleError, e:
            print("Error unpickling world: %s" % e)
        return restored_agent
