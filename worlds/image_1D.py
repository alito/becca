
import agent.viz_utils as viz_utils
from worlds.base_world import World as BaseWorld
import worlds.world_utils as world_utils

""" The Python Image Library, required by this world, is installed
as part of pyplot. This allows the loading and interpreting of .jpgs
"""
import matplotlib.pyplot as plt
import numpy as np

class World(BaseWorld):
    """ Image_1D, one-dimensional visual servo task
    In this task, BECCA can direct its gaze left and right along a
    mural. It is rewarded for directing it near the center. The
    mural is not represented using basic features, but rather using
    raw inputs, which BECCA must build into features. For a full writeup see:
    http://www.sandia.gov/~brrohre/doc/Rohrer11ImplementedArchitectureFeature.pdf
    Optimal performance is a reward of somewhere around 90 per time step.
    """
    def __init__(self):
        super(World, self).__init__()

        self.REPORTING_PERIOD = 10 ** 4       
        self.FEATURE_DISPLAY_INTERVAL = 10 ** 6
        self.LIFESPAN = 10 ** 5
        self.REWARD_MAGNITUDE = 100.
        self.JUMP_FRACTION = 0.01
        self.ANIMATE_PERIOD = 10 ** 3
        self.STEP_COST =  0.1 * self.REWARD_MAGNITUDE
        self.animate = False
        self.graphing = False
        self.name = 'one dimensional visual world'
        self.name_short = 'image_1D'
        self.announce()

        self.step_counter = 0
        self.fov_span = 10 
        self.num_sensors = 2 * self.fov_span ** 2
        self.num_primitives = 0
        self.num_actions = 9
        self.MAX_NUM_FEATURES = 50
        
        """ Initialize the image to be used as the environment """
        self.block_image_filename = "./images/bar_test.png" 
        self.data = plt.imread(self.block_image_filename)
        
        """ Convert it to grayscale if it's in color """
        if self.data.shape[2] == 3:
            """ Collapse the three RGB matrices into one black/white value matrix """
            self.data = np.sum(self.data, axis=2) / 3.0

        """ Define the size of the field of view, its range of allowable positions, 
        and its initial position.
        """
        image_width = self.data.shape[1]
        self.MAX_STEP_SIZE = image_width / 2
        self.TARGET_COLUMN = image_width / 2
        self.REWARD_REGION_WIDTH = image_width / 16
        self.NOISE_MAGNITUDE = 0.1
        
        self.column_history = []
        self.fov_height = np.min(self.data.shape)
        self.fov_width = self.fov_height
        self.column_min = np.ceil(self.fov_width / 2)
        self.column_max = np.floor(self.data.shape[1] - self.column_min)
        self.column_position = np.random.random_integers(self.column_min, self.column_max)
        self.block_width = self.fov_width / (self.fov_span + 2)

        self.sensors = np.zeros(self.num_sensors)
        self.primitives = np.zeros(self.num_primitives)
        self.last_feature_vizualized = 0


    def step(self, action): 
        self.timestep += 1
        
        """ Actions 0-3 move the field of view to a higher-numbered row (downward in the image) 
        with varying magnitudes, and actions 4-7 do the opposite.
        """
        column_step = np.round(action[0] * self.MAX_STEP_SIZE / 2 + 
                               action[1] * self.MAX_STEP_SIZE / 4 + 
                               action[2] * self.MAX_STEP_SIZE / 8 + 
                               action[3] * self.MAX_STEP_SIZE / 16 - 
                               action[4] * self.MAX_STEP_SIZE / 2 - 
                               action[5] * self.MAX_STEP_SIZE / 4 - 
                               action[6] * self.MAX_STEP_SIZE / 8 - 
                               action[7] * self.MAX_STEP_SIZE / 16)
        
        column_step = np.round( column_step * ( 1 + self.NOISE_MAGNITUDE * np.random.random_sample() * 2.0 - 
                                self.NOISE_MAGNITUDE * np.random.random_sample() * 2.0))
        self.column_position = self.column_position + int(column_step)
        self.column_position = max(self.column_position, self.column_min)
        self.column_position = min(self.column_position, self.column_max)

        """ At random intervals, jump to a random position in the world """
        if np.random.random_sample() < self.JUMP_FRACTION:
            self.column_position = np.random.random_integers(self.column_min, self.column_max)

        """ Create the sensory input vector """
        fov = self.data[:, self.column_position - self.fov_width / 2: 
                           self.column_position + self.fov_width / 2]
        center_surround_pixels = world_utils.center_surround( \
                        fov, self.fov_span, self.block_width, self.block_width)
        unsplit_sensors = center_surround_pixels.ravel()        
        sensors = np.concatenate((np.maximum(unsplit_sensors, 0), np.abs(np.minimum(unsplit_sensors, 0)) ))

        """ Calculate the reward """
        reward = 0
        if np.abs(self.column_position - self.TARGET_COLUMN) < \
            self.REWARD_REGION_WIDTH / 2.0:
            reward += self.REWARD_MAGNITUDE
        reward -= np.abs(column_step) / self.MAX_STEP_SIZE * self.STEP_COST
        
        if self.animate:
            print 'column_position: ', self.column_position, '  reward: ', reward[0]
        self.log(sensors, self.primitives, reward)
        return sensors, self.primitives, reward
    
            
    def log(self, sensors, primitives, reward):
        if self.graphing:
            self.column_history.append(self.column_position)
        self.display()

        if self.animate and (self.timestep % self.ANIMATE_PERIOD) == 0:
            plt.figure("Image sensed")
            sensed_image = np.reshape(0.5 * (sensors[:len(sensors)/2] - sensors[len(sensors)/2:] + 1), 
                                      (self.fov_span, self.fov_span))
            plt.gray()
            plt.imshow(sensed_image, interpolation='nearest')
            viz_utils.force_redraw()
            

    def set_agent_parameters(self, agent):
        agent.actor.reward_min = 0.
        agent.actor.reward_max = 100.
        
        """ These parameters create a very neat feature set and are good for
        long-term performance.
        """ 
        '''agent.perceiver.DISSIPATION_FACTOR = 3.0
        agent.perceiver.NEW_FEATURE_THRESHOLD = 0.05
        agent.perceiver.MIN_SIG_COACTIVITY = 0.995  * agent.perceiver.NEW_FEATURE_THRESHOLD
        agent.perceiver.PLASTICITY_UPDATE_RATE = agent.perceiver.NEW_FEATURE_THRESHOLD * 0.003
        '''
        return
            
         
    def display(self):
        if (self.timestep % self.REPORTING_PERIOD) == 0:
            print("world is %s timesteps old" % self.timestep)
            
            if self.graphing:
                plt.figure("Column history")
                plt.clf()
                plt.plot( self.column_history, 'k.')    
                plt.xlabel('time step')
                plt.ylabel('position (pixels)')
                plt.draw()
                viz_utils.force_redraw()
            return
        
    
    def is_time_to_display(self):
        if (self.timestep % self.FEATURE_DISPLAY_INTERVAL == 0):
            return True
        else:
            return False
        
    
    def vizualize_feature_set(self, feature_set):
        """ Provide an intuitive display of the features created by the agent """
        world_utils.vizualize_pixel_array_feature_set(feature_set, start=self.last_feature_vizualized, 
                                          world_name=self.name_short, save_eps=True, save_jpg=True)
        self.last_feature_vizualized = feature_set.shape[0]
        