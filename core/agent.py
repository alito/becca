import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np

from block import Block
from hub import Hub
import tools

class Agent(object):
    """ 
    A general reinforcement learning agent
    
    Takes in a time series of sensory input vectors and 
    a scalar reward and puts out a time series of action commands."""
    def __init__(self, num_sensors, num_actions, show=True, 
                 agent_name='test_agent'):
        """
        Configure the Agent

        num_sensors and num_actions are the only absolutely necessary
        arguments. They define the number of elements in the 
        sensors and actions arrays that the agent and the world use to
        communicate with each other. 
        """
        self.BACKUP_PERIOD = 10 ** 4
        self.show = show
        self.name = agent_name
        self.pickle_filename ="log/" + agent_name + ".pickle"
        # TODO: Automatically adapt to the number of sensors pass in
        self.num_sensors = num_sensors
        self.num_actions = num_actions

        # Initialize agent infrastructure
        self.num_blocks =  1
        first_block_name = ''.join(('block_', str(self.num_blocks - 1)))
        self.blocks = [Block(self.num_actions + self.num_sensors, 
                             name=first_block_name)]
        self.hub = Hub(self.blocks[0].max_cables)
        self.action = np.zeros((self.num_actions,1))
        self.cumulative_reward = 0
        self.time_since_reward_log = 0 
        self.reward_history = []
        self.reward_steps = []
        self.surprise_history = []
        self.recent_surprise_history = [0.] * 100
        self.timestep = 0
        self.graphing = True

    def step(self, sensors, reward):
        """ Step through one time interval of the agent's operation """
        self.timestep += 1
        if sensors.ndim == 1:
            sensors = sensors[:,np.newaxis]
        self.reward = reward
        # Propogate the new sensor inputs up through the blocks
        cable_activities = np.vstack((self.action, sensors))
        for block in self.blocks:
            cable_activities = block.step_up(cable_activities) 
        # Create a new block if the top block has had enough bundles assigned
        block_bundles_full = (float(block.bundles_created()) / 
                              float(block.max_bundles))
        block_initialization_threshold = .5
        if block_bundles_full > block_initialization_threshold:
            self.num_blocks +=  1
            next_block_name = ''.join(('block_', str(self.num_blocks - 1)))
            self.blocks.append(Block(self.num_actions + self.num_sensors,
                                     name=next_block_name, 
                                     level=self.num_blocks))
            cable_activities = self.blocks[-1].step_up(cable_activities) 
            self.hub.add_cables(self.blocks[-1].max_cables)
            print "Added block", self.num_blocks - 1

        self.hub.step(self.blocks, self.reward) 
        
        # Propogate the deliberation_goal_votes down through the blocks
        # debug
        agent_surprise = 0.0
        cable_goals = np.zeros((cable_activities.size,1))
       
        for block in reversed(self.blocks):
            cable_goals = block.step_down(cable_goals)
            if np.nonzero(block.surprise)[0].size > 0:
                agent_surprise = np.sum(block.surprise)
        self.recent_surprise_history.pop(0)
        self.recent_surprise_history.append(agent_surprise)
        self.typical_surprise = np.median(np.array(
                self.recent_surprise_history))
        mod_surprise = agent_surprise - self.typical_surprise
        self.surprise_history.append(mod_surprise)

        # Strip the actions off the deliberation_goal_votes to make 
        # the current set of actions.
        # For actions, each goal is a probability threshold. If a roll of
        # dice comes up lower than the goal value, the action is taken
        # with a magnitude of 1.
        self.action = cable_goals[:self.num_actions,:] 
        if (self.timestep % self.BACKUP_PERIOD) == 0:
                self._save()    
        # Log reward
        self.cumulative_reward += reward
        self.time_since_reward_log += 1
        # debug
        if np.random.random_sample() < 0.001:
            self.visualize()
        return self.action

    def get_index_projections(self, to_screen=False):
        """
        Get representations of all the bundles in each block 
        
        Every feature is projected down through its own block and
        the blocks below it until its cable_contributions on sensor inputs 
        and actions is obtained. This is a way to represent the
        receptive field of each feature.

        Returns a list containing the cable_contributions for each feature 
        in each block.
        """
        all_projections = []
        all_bundle_activities = []
        for block_index in range(len(self.blocks)):
            block_projections = []
            block_bundle_activities = []
            num_bundles = self.blocks[block_index].max_bundles
            for bundle_index in range(num_bundles):    
                bundles = np.zeros((num_bundles, 1))
                bundles[bundle_index, 0] = 1.
                cable_contributions = self._get_index_projection(
                        block_index,bundles)
                if np.nonzero(cable_contributions)[0].size > 0:
                    block_projections.append(cable_contributions)
                    block_bundle_activities.append(self.blocks[block_index].
                            bundle_activities[bundle_index])
                    # Display the cable_contributions in text form if desired
                    if to_screen:
                        print 'cable_contributions', \
                            self.blocks[block_index].name, \
                            'feature', bundle_index
                        for i in range(cable_contributions.shape[1]):
                            print np.nonzero(cable_contributions)[0][
                                    np.where(np.nonzero(
                                    cable_contributions)[1] == i)]
            if len(block_projections) > 0:
                all_projections.append(block_projections)
                all_bundle_activities.append(block_bundle_activities)
        return (all_projections, all_bundle_activities)
  
    def _get_index_projection(self, block_index, bundles):
        """
        Get the cable_contributions for bundles
        
        Recursively project bundles down through blocks
        until the bottom block is reached. Feature values is a 
        two-dimensional array and can contain
        several columns. Each column represents a state, and their
        order represents a temporal progression. During cable_contributions
        to the next lowest block, the number of states
        increases by one. 
        
        Return the cable_contributions in terms of basic sensor 
        inputs and actions. 
        """
        if block_index == -1:
            return bundles
        cable_contributions = np.zeros((self.blocks[block_index].max_cables, 
                               bundles.shape[1] + 1))
        for bundle_index in range(bundles.shape[0]):
            for time_index in range(bundles.shape[1]):
                if bundles[bundle_index, time_index] > 0:
                    new_contribution = self.blocks[
                            block_index].get_index_projection(bundle_index)
                    cable_contributions[:,time_index:time_index + 2] = (
                            np.maximum(
                            cable_contributions[:,time_index:time_index + 2], 
                            new_contribution))
        cable_contributions = self._get_index_projection(block_index - 1, 
                                                   cable_contributions)
        return cable_contributions

    def visualize(self):
        """ Show the current state and some history of the agent """
        print ' '.join([self.name, 'is', str(self.timestep), 'time steps old'])
        self.reward_history.append(float(self.cumulative_reward) / 
                                   (self.time_since_reward_log + 1))
        self.cumulative_reward = 0    
        self.time_since_reward_log = 0
        self.reward_steps.append(self.timestep)
        self._show_reward_history()
        for block in self.blocks:
            block.visualize()
            pass
        return
 
    def report_performance(self):
        """ Report on the reward amassed by the agent """
        performance = np.mean(self.reward_history)
        print("Final performance is %f" % performance)
        self._show_reward_history(hold_plot=self.show)
        return performance
    
    def _show_reward_history(self, hold_plot=False, 
                            filename='log/reward_history.png'):
        """ Show the agent's reward history and save it to a file """
        if self.graphing:
            fig = plt.figure(1)
            plt.plot(self.reward_steps, self.reward_history)
            plt.xlabel("time step")
            plt.ylabel("average reward")
            plt.title(''.join(('Reward history for ', self.name)))
            fig.show()
            fig.canvas.draw()
            plt.savefig(filename, format='png')
            if hold_plot:
                plt.show()
        return
    
    def _save(self):
        """ Archive a copy of the agent object for future use """
        success = False
        make_backup = True
        print "Attempting to save agent..."
        try:
            with open(self.pickle_filename, 'wb') as agent_data:
                pickle.dump(self, agent_data)
            if make_backup:
                with open(''.join((self.pickle_filename, '.bak')), 
                          'wb') as agent_data_bak:
                    pickle.dump(self, agent_data_bak)
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
                print(''.join(('Agent restored at timestep ', 
                               str(loaded_agent.timestep),
                               ' from ', self.pickle_filename)))
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
