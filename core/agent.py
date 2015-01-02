""" the Agent class """
import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import os
mod_path = os.path.dirname(os.path.abspath(__file__))
import arborkey
import drivetrain
import hub
import mainspring
import spindle
import tools

class Agent(object):
    """ 
    A general reinforcement learning agent
    
    It takes in an array of sensor values and 
    a reward and puts out an array of action commands at each time step.
    """
    def __init__(self, num_sensors, num_actions, show=True, 
                 agent_name='test_agent'):
        """
        Configure the Agent

        num_sensors and num_actions are the only absolutely necessary
        arguments. They define the number of elements in the 
        sensors and actions arrays that the agent and the world use to
        communicate with each other. 
        """
        self.BACKUP_PERIOD = 1e4
        self.FORGETTING_RATE = 1e-3
        self.show = show
        self.name = agent_name
        self.log_dir = os.path.normpath(os.path.join(mod_path, '..', 'log'))
        if not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir)
        self.pickle_filename = os.path.join(
                self.log_dir, '.'.join([agent_name, 'pickle']))
        self.num_sensors = num_sensors
        self.num_actions = num_actions

        # Initialize agent infrastructure.
        # Choose min_cables to account for all sensors, actions, 
        # and two reward sensors, at a minimum.
        min_cables = self.num_actions + self.num_sensors
        self.drivetrain = drivetrain.Drivetrain(min_cables)
        num_cables = self.drivetrain.cables_per_gearbox
        self.hub = hub.Hub(num_cables)
        self.spindle = spindle.Spindle(num_cables)
        self.mainspring = mainspring.Mainspring(num_cables)
        self.arborkey = arborkey.Arborkey()
        self.action = np.zeros((self.num_actions,1))
        self.cumulative_reward = 0
        self.time_since_reward_log = 0 
        self.reward_history = []
        self.reward_steps = []
        self.reward_max = -tools.BIG
        self.timestep = 0
        self.graphing = True

    def step(self, sensors, unscaled_reward):
        # Adapt the reward so that it falls between -1 and 1 
        self.reward_max = np.maximum(np.abs(unscaled_reward), self.reward_max)
        self.raw_reward = unscaled_reward / (self.reward_max + tools.EPSILON)
        self.reward_max *= (1. - self.FORGETTING_RATE)
        self.timestep += 1
        if sensors.ndim == 1:
            sensors = sensors[:,np.newaxis]
        # Propogate the new sensor inputs through the drivetrain
        feature_activities = self.drivetrain.step_up(self.action, sensors)
        # The drivetrain will grow over time as the agent gains experience.
        # If the drivetrain added a new gearbox, scale the hub up appropriately.
        if self.drivetrain.gearbox_added:
            self.hub.add_cables(self.drivetrain.cables_per_gearbox)
            self.spindle.add_cables(self.drivetrain.cables_per_gearbox)
            self.mainspring.add_cables(self.drivetrain.cables_per_gearbox)
            self.arborkey.add_cables(self.drivetrain.cables_per_gearbox)
            self.drivetrain.gearbox_added = False
        # Feed the feature_activities to the hub for calculating goals
        hub_goal, hub_reward = self.hub.step(feature_activities, 
                                             self.raw_reward) 
        # Evaluate the goal using the mainspring
        mainspring_reward = self.mainspring.evaluate(hub_goal) 
        # Choose a single feature to attend 
        (attended_index, attended_activity) = self.spindle.step(
                feature_activities)
        # Incorporate the intended feature into short- and long-term memory
        self.mainspring.step(attended_index, attended_activity, 
                             self.raw_reward)
        # Pass the hub goal on to the arborkey for further evaluation
        goal_cable = self.arborkey.step(hub_goal, mainspring_reward, 
                                        self.raw_reward)
        self.hub.update(feature_activities, goal_cable)
        if goal_cable is not None:
            self.drivetrain.assign_goal(goal_cable)
        self.action = self.drivetrain.step_down()
        # debug: Choose a single random action 
        random_action = False
        if random_action:
            self.action = np.zeros(self.action.shape)
            random_action_index = np.random.randint(self.action.size)
            self.action[random_action_index] = 1. 

        if (self.timestep % self.BACKUP_PERIOD) == 0:
                self._save()    
        # Log reward
        self.cumulative_reward += unscaled_reward
        self.time_since_reward_log += 1
        # debug
        if np.random.random_sample() < 0.001:
            self.visualize()
        return self.action

    def get_index_projections(self, to_screen=False):
        return self.drivetrain.get_index_projections(to_screen=to_screen)

    def visualize(self):
        """ Show the current state and some history of the agent """
        print ' '.join([self.name, 'is', str(self.timestep), 'time steps old'])
        self.reward_history.append(float(self.cumulative_reward) / 
                                   (self.time_since_reward_log + 1))
        self.cumulative_reward = 0    
        self.time_since_reward_log = 0
        self.reward_steps.append(self.timestep)
        self._show_reward_history()

        self.drivetrain.visualize()
        self.spindle.visualize()
        #self.hub.visualize()
        #self.mainspring.visualize()
        self.arborkey.visualize()
 
    def report_performance(self):
        performance = np.mean(self.reward_history)
        print("Final performance is %f" % performance)
        self._show_reward_history(hold_plot=self.show)
        return performance
    
    def _show_reward_history(self, hold_plot=False, 
                            filename=None):
        """ Show the agent's reward history and save it to a file """
        if self.graphing:
            fig = plt.figure(1)
            plt.plot(self.reward_steps, self.reward_history)
            plt.xlabel("time step")
            plt.ylabel("average reward")
            plt.title(''.join(('Reward history for ', self.name)))
            fig.show()
            fig.canvas.draw()
            if filename is None:
                filename = os.path.join(self.log_dir, 'reward_history.png')
            plt.savefig(filename, format='png')
            if hold_plot:
                plt.show()
    
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
