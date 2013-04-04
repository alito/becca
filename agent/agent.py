
from cog import Cog
from level import Level
import utils as ut

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
        
        self.timestep = 0
        self.graphing = True
        
        self.cumulative_reward = 0
        self.reward_history = []
        self.reward_steps = []
        
        self.num_levels =  5
        self.levels = []
        for level_ctr in range(self.num_levels):
            self.levels.append(Level(name='level'+str(level_ctr)))        

        self.REWARD_RANGE_DECAY_RATE = 10 ** -10
        self.reward_min = ut.BIG
        self.reward_max = -ut.BIG
       
        self.SENSOR_RANGE_DECAY_RATE = 10 ** -3
        self.sensor_min = np.ones((self.num_sensors, 1)) * ut.BIG
        self.sensor_max = np.ones((self.num_sensors, 1)) * (-ut.BIG)
        
        
    def step(self, raw_sensors, raw_reward):
        self.timestep += 1
        
        """ Modify the reward so that it automatically falls between 0 and 1 """        
        self.reward_min = np.minimum(raw_reward, self.reward_min)
        self.reward_max = np.maximum(raw_reward, self.reward_max)
        spread = self.reward_max - self.reward_min
        self.reward = (raw_reward - self.reward_min) / (spread + ut.EPSILON)
        self.reward_min += spread * self.REWARD_RANGE_DECAY_RATE
        self.reward_max -= spread * self.REWARD_RANGE_DECAY_RATE
                
        if raw_sensors.ndim == 1:
            raw_sensors = raw_sensors[:,np.newaxis]
        
        """ Modify sensor inputs so that they fall between 0 and 1 """
        self.sensor_min = np.minimum(raw_sensors , self.sensor_min)
        self.sensor_max = np.maximum(raw_sensors , self.sensor_max)
        spread = self.sensor_max - self.sensor_min
        sensors = (raw_sensors - self.sensor_min) / (spread + ut.EPSILON)
        self.sensor_min += spread * self.SENSOR_RANGE_DECAY_RATE
        self.sensor_max -= spread * self.SENSOR_RANGE_DECAY_RATE
        
        # propogate the new information up and down through the levels
        feature_inputs = np.vstack((self.action, sensors))
        for level in self.levels:
            feature_inputs = level.step_up(feature_inputs, self.reward) 
        #if feature_inputs.size > 0:
        #    print 'Consider creating a new level'
        goals = np.zeros((feature_inputs.size,1))
        for level in reversed(self.levels):
            goals = level.step_down(goals) 
            print 'surprise', level.surprise.ravel()
        # Strip the actions off the goals to make the current set of actions
        if goals.size < self.num_actions:
            goals = ut.pad(goals,(self.num_actions, 0))
        self.action = np.zeros((self.num_actions, 1))
        action_thresholds = np.random.random_sample((self.num_actions, 1))
        self.action[np.nonzero(goals[:self.num_actions,:] 
                    > action_thresholds)] = 1.
        self.log(raw_reward)
        return self.action

    def get_projections(self, to_screen=False):
        all_projections = []
	for level_index in range(len(self.levels)):
            level_projections = []
            num_features = self.levels[level_index].output_map.shape[0]
            for feature_index in range(num_features):    
                features = np.zeros((num_features, 1))
                features[feature_index, 0] = 1.
                projection = self.get_projection(level_index, features)
                level_projections.append(projection)
                if to_screen:
                    print 'projection', self.levels[level_index].name, 'feature', feature_index
                    for i in range(projection.shape[1]):
                        print np.nonzero(projection)[0][np.where(np.nonzero(projection)[1] == i)]
            if len(level_projections) > 0:
                all_projections.append(level_projections)
        return all_projections
  
    def get_projection(self, level_index, features):
        if level_index == -1:
            return features
        projection = np.zeros((self.levels[level_index].num_feature_inputs, features.shape[1] + 1))
        for feature_index in range(features.shape[0]):
            for state_index in range(features.shape[1]):
                if features[feature_index,state_index] > 0:
                    projection[:,state_index:state_index + 2] = \
                                np.maximum(projection[:,state_index:state_index + 2], 
                                self.levels[level_index].get_projection(feature_index) )
        projection = self.get_projection(level_index - 1, projection)            
        return projection

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
            #for level in self.levels:
            #    level.display()
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
