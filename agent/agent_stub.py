""" This is a base class class, used for debugging and testing the world """

import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np
#from utils import force_redraw

class AgentStub(object):
    
    def __init__(self, agent_name, num_sensors, num_primitives, 
                 num_actions, max_num_features=None):
        
        self.pickle_filename = agent_name + "_agent.pickle"
        
        self.REPORTING_PERIOD = 10 ** 3
        self.BACKUP_PERIOD = 2* 10 ** 3

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
            self.record_reward_history()
            self.show_reward_history()
            self.cumulative_reward = 0    
            print("Timesteps: %s" % self.timestep)

        if (self.timestep % self.BACKUP_PERIOD) == 0:
            self.save()    

        
    def step(self, sensors, primitives, reward):
        self.timestep += 1   

        self.reward = reward
        self.actions = np.zeros(self.num_actions)
        self.actions[np.random.randint(self.num_actions)] = 1
        
        self.log()

        return self.actions

        
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
        