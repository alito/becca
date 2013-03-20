
from cog import Cog
from level import Level
import utils

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
    def __init__(self, num_sensors, num_actions, max_num_features=1000, show=True, agent_name='my'):
        
        self.show = show
        self.pickle_filename ="log/" + agent_name + "_agent.pickle"
        
        self.REPORTING_PERIOD = 10 ** 3
        self.BACKUP_PERIOD = 10 ** 8

        self.num_sensors = num_sensors
        self.num_actions = num_actions

        self.reward = 0
        self.reward2 = 0
        self.action = np.zeros((self.num_actions,1))
        self.action_last = np.zeros((self.num_actions,1))
        
        self.timestep = 0
        self.graphing = True
        
        self.cumulative_reward = 0
        self.reward_history = []
        self.reward_steps = []
        
        self.two_level = True
        if self.two_level:
            level1_cog_out_size = 30#1. * num_sensors
            num_level2_features_in = level1_cog_out_size + num_actions
            level1_cog_in_size = num_sensors
        else:
            num_level2_features_in = 0
            level1_cog_in_size = num_sensors + num_actions
            level1_cog_out_size = 0
            
        """ Construct the hierarchy """
        level1_cogs = []
        level1_cogs.append(Cog(level1_cog_in_size, level1_cog_out_size, name='cog1'))
        self.level1 = Level(level1_cogs, name='level1')
        self.level1_goal = np.zeros((level1_cog_out_size, 1))
        
        level2_cogs = []
        level2_cog_in_size = num_level2_features_in
        level2_cog_out_size = 0
        level2_cogs.append(Cog(level2_cog_in_size, level2_cog_out_size, name='cog2'))
        self.level2 = Level(level2_cogs, name='level2')
        self.level2_features = np.zeros((level2_cog_in_size, 1))
 
        self.REWARD_RANGE_DECAY_RATE = 10 ** -10
        self.reward_min = utils.BIG
        self.reward_max = -utils.BIG
       
        self.SENSOR_RANGE_DECAY_RATE = 10 ** -3
        self.sensor_min = np.ones((self.num_sensors, 1)) * utils.BIG
        self.sensor_max = np.ones((self.num_sensors, 1)) * (-utils.BIG)
        
        
    def step(self, raw_sensors, raw_reward):
        self.timestep += 1
        
        """ Delay by one time step """
        self.reward2 = self.reward
        
        """ Modify the reward so that it automatically falls between 0 and 1 """        
        self.reward_min = np.minimum(raw_reward, self.reward_min)
        self.reward_max = np.maximum(raw_reward, self.reward_max)
        spread = self.reward_max - self.reward_min
        self.reward = (raw_reward - self.reward_min) / (spread + utils.EPSILON)
        self.reward_min += spread * self.REWARD_RANGE_DECAY_RATE
        self.reward_max -= spread * self.REWARD_RANGE_DECAY_RATE
                
        if raw_sensors.ndim == 1:
            raw_sensors = raw_sensors[:,np.newaxis]
        
        """ Modify sensor inputs so that they fall between 0 and 1 """
        self.sensor_min = np.minimum(raw_sensors , self.sensor_min)
        self.sensor_max = np.maximum(raw_sensors , self.sensor_max)
        spread = self.sensor_max - self.sensor_min
        sensors = (raw_sensors - self.sensor_min) / (spread + utils.EPSILON)
        self.sensor_min += spread * self.SENSOR_RANGE_DECAY_RATE
        self.sensor_max -= spread * self.SENSOR_RANGE_DECAY_RATE
        
        if self.two_level:
            new_level2_features = self.level1.step_up(sensors, self.reward) 
            self.level2.step_up(np.vstack((self.action, new_level2_features)), self.reward) 
            level2_feedback= self.level2.step_down() 
            
            """ Strip the actions off the goals to make the current set of actions """
            self.action_last = self.action.copy()
            self.action = np.sign(level2_feedback[:self.num_actions:,:])
            self.level1_goal = level2_feedback[self.num_actions:,:]
            
            self.goal = self.level1.step_down(self.level1_goal) 
            '''self.goal, new_level2_features = self.level1.step(sensors, self.reward, hi_goal=self.level1_goal) 
            level2_feedback, dum = self.level2.step(np.vstack((self.action, self.level2_features)),
                                                    self.reward2) 
            self.level2_features = new_level2_features
            '''
            
        else:
            self.goal, new_level2_features = self.level1.step(np.vstack((sensors, self.action)), self.reward) 
        
            """ Strip the actions off the goals to make the current set of actions """
            self.action = np.sign(self.goal[self.num_sensors: self.num_sensors + self.num_actions,:])
        
        self.log(raw_reward)
        return self.action


    def get_projections(self):
        return self.level1.get_projections()
    
    def log(self, raw_reward):
        self.cumulative_reward += raw_reward

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
            self.level1.display()
            if self.two_level:
                self.level2.display()
                
            feature_projections = self.get_projections()  

        return
 
    
    def report_performance(self):
        performance = np.mean(self.reward_history)
        print("Final performance is %f" % performance)
        self.show_reward_history(save_eps=True, block=self.show)
        return performance
        
    
    def show_reward_history(self, block=False, save_eps=False,
                            epsfilename='log/reward_history.eps'):
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
