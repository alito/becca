
from grouper import Grouper
from learner import Learner
import utils

import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np


class Agent(object):
    """ A general reinforcement learning agent, modeled on observations and 
    theories of human performance. It takes in a time series of 
    sensory input vectors and a scalar reward and puts out a time series 
    of motor commands. New features are created as necessary to adequately 
    represent the data.
    """
    def __init__(self, agent_name, num_sensors, num_primitives, 
                 num_actions, max_num_features=1000):
        
        self.pickle_filename = agent_name + "_agent.pickle"
        
        self.REPORTING_PERIOD = 10 ** 3
        self.BACKUP_PERIOD = 10 ** 3

        self.num_sensors = num_sensors
        self.num_primitives = num_primitives
        self.num_actions = num_actions

        self.reward = 0
        self.actions = np.zeros(self.num_actions)
        
        self.timestep = 0
        self.graphing = True
        
        self.cumulative_reward = 0
        self.reward_history = []
        self.reward_steps = []
        
        self.grouper = Grouper( num_sensors, num_actions, num_primitives, 
                                max_num_features)
        self.learner = Learner( num_sensors, num_actions, num_primitives)

               
    def restore(self):
        
        restored_agent = self
        try:
            with open(self.pickle_filename, 'rb') as agent_data:
                loaded_agent = pickle.load(agent_data)

            """ Compare the number of channels in the restored agent with
            those in the already initialized agent. 
            If it matches, accept the agent. If it doesn't,
            print a message, and keep the just-initialized agent.
            """
            if((loaded_agent.num_sensors == self.num_sensors) & 
               (loaded_agent.num_primitives == self.num_primitives) & 
               (loaded_agent.num_actions == self.num_actions)):
            
                print("Agent restored at timestep " + str(loaded_agent.timestep))
                restored_agent = loaded_agent

            else:
                print("The agent " + self.pickle_filename + " does not have " +
                      "the same number of input and output elements " +
                      "as the world.")
                print("Creating a new agent from scratch.")
            
        except IOError:
            print("Couldn't open %s for loading" % self.pickle_filename)
        except pickle.PickleError, e:
            print("Error unpickling world: %s" % e)

        return restored_agent

    
    def save(self):
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


    def record_reward_history(self):
        self.reward_history.append(float(self.cumulative_reward) / 
                                   self.REPORTING_PERIOD)
        self.reward_steps.append(self.timestep)
                    
            
    def show_reward_history(self, show=False):
        if self.graphing:
            plt.figure("reward history")
            plt.plot(self.reward_steps, self.reward_history)
            plt.xlabel("time step")
            plt.ylabel("average reward")
            plt.draw()
            #force_redraw()
            
            if show:
                plt.show()
            
        
    def log(self):
        """ Logs the state of the world into a history that can be used to
        evaluate and understand Becca's behavior
        """
        self.cumulative_reward += self.reward

        if (self.timestep % self.REPORTING_PERIOD) == 0:
            self.display()
            self.cumulative_reward = 0    


        if (self.timestep % self.BACKUP_PERIOD) == 0:
            self.save()    


    def display(self):
        #print self.timestep
        if (self.timestep % self.REPORTING_PERIOD) == 0:
            self.record_reward_history()
            self.show_reward_history()
            print("agent is %s timesteps old" % self.timestep)
            print("%s inputs total" % self.grouper.n_inputs)
            
            self.grouper.visualize()
            self.learner.visualize()
            utils.force_redraw()

        
    def step(self, sensors, primitives, reward):
        """ Advances the agent's operation by one time step """
        
        self.timestep += 1

        self.sensors = sensors
        self.primitives = primitives
        self.reward = reward

        """
        Feature creator
        ======================================================
        """
        feature_activity = self.grouper.step(sensors, 
                                             primitives, 
                                             self.actions)
        
        """
        Reinforcement learner
        ======================================================
        """
        self.actions = self.learner.step(feature_activity) 

        return
        
        
    def report_performance(self):
        """ When the world terminates, this returns the performance 
        of the agent, a real value between -1 and 1. Before reaching
        the termination condition, it returns a value less than -1.
        Any terminating activities or reports should be included
        in this method too.
        """
        performance = (np.mean(self.reward_history[-3:]) / 3)
        print("Final performance is %f" % performance)
        
        return performance
        