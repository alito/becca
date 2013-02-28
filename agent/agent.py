
from actor import Actor
from perceiver import Perceiver
import viz_utils

import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np

class Agent(object):
    """ A general reinforcement learning agent, modeled on observations and 
    theories of human performance. It takes in a time series of 
    sensory input vectors and a scalar reward and puts out a time series 
    of action commands. New features are created as necessary to adequately 
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
        self.action = np.zeros((self.num_actions,1))
        
        self.timestep = 0
        self.graphing = True
        
        self.cumulative_reward = 0
        self.reward_history = []
        self.reward_steps = []
        
        self.perceiver = Perceiver(num_sensors, num_primitives, num_actions, 
                                max_num_features)
        self.actor = Actor(num_primitives, num_actions, max_num_features)
 
        
    def step(self, sensors, primitives, reward):
        self.timestep += 1
        self.reward = reward
        
        """ Feature extractor """
        feature_activity, n_features = self.perceiver.step(sensors, primitives, self.action)
        
        """ Reinforcement learner """
        self.action = self.actor.step(feature_activity, reward, n_features) 
        
        self.log()
        return self.action

    
    def log(self):
        self.cumulative_reward += self.reward

        if (self.timestep % self.REPORTING_PERIOD) == 0:
            self.display()
            self.cumulative_reward = 0    

        if (self.timestep % self.BACKUP_PERIOD) == 0:
            self.save()    
        return

    
    def display(self):
        if (self.timestep % self.REPORTING_PERIOD) == 0:
            print self.timestep, 'time steps'
            self.reward_history.append(float(self.cumulative_reward) / self.REPORTING_PERIOD)
            self.reward_steps.append(self.timestep)
            self.show_reward_history(save_eps=True)
            #self.perceiver.visualize(save_eps=True)
            #self.actor.visualize()
        return
 
    
    def report_performance(self, show=True):
        performance = np.mean(self.reward_history)
        print("Final performance is %f" % performance)
        self.show_reward_history(save_eps=True)
        if show:
            plt.show()    
        return performance
        
    
    def show_reward_history(self, show=False, save_eps=False,
                            epsfilename='log/reward_history.eps'):
        if self.graphing:
            plt.figure(1)
            plt.plot(self.reward_steps, self.reward_history)
            plt.xlabel("time step")
            plt.ylabel("average reward")
            viz_utils.force_redraw()
            if save_eps:
                plt.savefig(epsfilename, format='eps')
            if show:
                plt.show()
        return
    
    
    def save(self):
        success = False
        try:
            with open(self.pickle_filename, 'wb') as agent_data:
                pickle.dump(self, agent_data)
            print("Agent data saved at " + str(self.timestep) + " time steps")
        except IOError as err:
            print("File error: " + str(err) + " encountered while saving agent data")
        except pickle.PickleError as perr: 
            print("Pickling error: " + str(perr) + " encountered while saving agent data")        
        else:
            success = True
        return success
           
        
    def restore(self):
        restored_agent = self
        try:
            with open(self.pickle_filename, 'rb') as agent_data:
                loaded_agent = pickle.load(agent_data)

            """ Compare the number of channels in the restored agent with those in the 
            already initialized agent. If it matches, accept the agent. If it doesn't,
            print a message, and keep the just-initialized agent.
            """
            if((loaded_agent.num_sensors == self.num_sensors) and 
               (loaded_agent.num_primitives == self.num_primitives) and
               (loaded_agent.num_actions == self.num_actions)):
            
                print("Agent restored at timestep " + 
                      str(loaded_agent.timestep))
                restored_agent = loaded_agent
            else:
                print("The agent " + self.pickle_filename + " does not have " +
                      "the same number of input and output elements as the world.")
                print("Creating a new agent from scratch.")
                
        except IOError:
            print("Couldn't open %s for loading" % self.pickle_filename)
        except pickle.PickleError, e:
            print("Error unpickling world: %s" % e)
        return restored_agent
