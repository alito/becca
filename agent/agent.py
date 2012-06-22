
from grouper import Grouper
from learner import Learner
import viz_utils

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
    def __init__(self, num_sensors, num_primitives, 
                 num_actions, max_num_features=1000, agent_name='my'):
        
        self.pickle_filename ="log/" + agent_name + "_agent.pickle"
        
        self.REPORTING_PERIOD = 10 ** 3
        self.BACKUP_PERIOD = 10 ** 8

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
        
        self.grouper = Grouper(num_sensors, num_primitives, num_actions, 
                                max_num_features)
        self.learner = Learner(num_primitives, num_actions)
 
        
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
        self.actions = self.learner.step(feature_activity, reward) 
        
        self.log()

        return self.actions

    
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
        if (self.timestep % self.REPORTING_PERIOD) == 0:
            self.record_reward_history()
            self.show_reward_history(save_eps=True)
            print "agent is ", self.timestep ," timesteps old" 
            print self.grouper.n_inputs , " inputs total"  
            print "Total size is about ", self.size() / 10 ** 6 , \
                    " million elements" 
            
            self.grouper.visualize(save_eps=True)
            #self.learner.visualize()
 
    
    def record_reward_history(self):
        self.reward_history.append(float(self.cumulative_reward) / 
                                   self.REPORTING_PERIOD)
        self.reward_steps.append(self.timestep)
                    
            
    def show_reward_history(self, show=False, save_eps=False,
                            epsfilename='log/reward_history.eps'):
        if self.graphing:
            fig = plt.figure("Reward history")
            plt.plot(self.reward_steps, self.reward_history)
            plt.xlabel("time step")
            plt.ylabel("average reward")
            viz_utils.force_redraw()

            if save_eps:
                plt.savefig(epsfilename, format='eps')

            if show:
                plt.show()
    
    
    def size(self):
        """ Determine the approximate number of elements being used by the
        class and its members. Created to debug an apparently excessive 
        use of memory.
        """
        total = 0
        total += self.grouper.size()
        total += self.learner.size()
        
        return total
            
        
    def report_performance(self, show=True):
        """ When the world terminates, this returns the performance 
        of the agent, a real value between -1 and 1. Before reaching
        the termination condition, it returns a value less than -1.
        Any terminating activities or reports should be included
        in this method too.
        """
        tail_length = int(np.ceil(len(self.reward_history) / 4))
        performance = np.mean(self.reward_history[-tail_length:])
        print("Final performance is %f" % performance)
        
        self.grouper.visualize(save_eps=True)
        self.learner.visualize(save_eps=True)
        self.show_reward_history(save_eps=True)

        if show:
            plt.show()    
        
        return performance
        
    
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
            if((loaded_agent.num_sensors == self.num_sensors) and 
               (loaded_agent.num_primitives == self.num_primitives) and
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


