
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
    Optimal performance is a reward of 100 per time step.
    Good performance for BECCA is a reward of somewhere over 70 per time step.
    """
    def __init__(self):
        super(World, self).__init__()

        self.REPORTING_PERIOD = 10 ** 4       
        self.FEATURE_DISPLAY_INTERVAL = 10 ** 6
        self.LIFESPAN = 2 * 10 ** 4
        self.REWARD_MAGNITUDE = 100.
        self.ANIMATE_PERIOD = 10 ** 2
        self.animate = False
        self.graphing = False
        self.name = 'one dimensional visual world'
        self.announce()

        self.step_counter = 0
        self.fov_span = 10 
        
        self.num_sensors = self.fov_span ** 2
        self.num_primitives = 0
        self.num_actions = 9

        self.column_history = []

        """ Initialize the image to be used as the environment """
        self.image_filename = "./images/bar_test.png" 
        self.data = plt.imread(self.image_filename)
        
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
        
        self.fov_height = np.min(self.data.shape)
        self.fov_width = self.fov_height
        self.column_min = np.ceil(self.fov_width / 2)
        self.column_max = np.floor(self.data.shape[1] - self.column_min)
        self.column_position = np.random.random_integers(self.column_min, self.column_max)
        self.block_width = self.fov_width / (self.fov_span + 2)

        self.sensors = np.zeros(self.num_sensors)
        self.primitives = np.zeros(self.num_primitives)


    def step(self, action): 
        self.timestep += 1
        
        """ Actions 0-3 move the field of view to a higher-numbered 
        row (downward in the image) with varying magnitudes, and
        actions 4-7 do the opposite.
        """
        column_step = np.round(action[0] * self.MAX_STEP_SIZE / 2 + 
                               action[1] * self.MAX_STEP_SIZE / 4 + 
                               action[2] * self.MAX_STEP_SIZE / 8 + 
                               action[3] * self.MAX_STEP_SIZE / 16 - 
                               action[4] * self.MAX_STEP_SIZE / 2 - 
                               action[5] * self.MAX_STEP_SIZE / 4 - 
                               action[6] * self.MAX_STEP_SIZE / 8 - 
                               action[7] * self.MAX_STEP_SIZE / 16)
        
        column_step = np.round( column_step * ( 1 + self.NOISE_MAGNITUDE * 
                                np.random.random_sample() * 2.0 - 
                                self.NOISE_MAGNITUDE * np.random.random_sample() * 2.0))
        
        self.column_position = self.column_position + int(column_step)

        self.column_position = max(self.column_position, self.column_min)
        self.column_position = min(self.column_position, self.column_max)

        """ Create the sensory input vector """
        fov = self.data[:, self.column_position - self.fov_width / 2: 
                           self.column_position + self.fov_width / 2]
        
        center_surround_pixels = world_utils.center_surround( \
                        fov, self.fov_span, self.block_width, self.block_width)

        sensors = center_surround_pixels.ravel()

        """ Calculate the reward """
        reward = 0
        if abs(self.column_position - self.TARGET_COLUMN) < \
            self.REWARD_REGION_WIDTH / 2.0:
            reward += self.REWARD_MAGNITUDE
        reward -= column_step / self.MAX_STEP_SIZE * 0.1 * self.REWARD_MAGNITUDE
        
        self.log(sensors, self.primitives, reward)
        return sensors, self.primitives, reward
    
            
    def log(self, sensors, primitives, reward):
        
        if self.graphing:
            self.column_history.append(self.column_position)

        self.display()

        if self.animate and (self.timestep % self.ANIMATE_PERIOD) == 0:
            plt.figure("Image sensed")
            sensed_image = np.reshape(sensors[:len(sensors)/2], 
                                      (self.fov_span, self.fov_span))
            plt.gray()
            plt.imshow(sensed_image, interpolation='nearest')
            viz_utils.force_redraw()
            
            #plt.show()


    def set_agent_parameters(self, agent):
        
        """ These parameters create a very neat feature set and are good for
        long-term performance.
        """ 
        '''agent.perceiver.DISSIPATION_FACTOR = 3.0
        agent.perceiver.NEW_FEATURE_THRESHOLD = 0.05
        agent.perceiver.MIN_SIG_COACTIVITY = 0.995  * agent.perceiver.NEW_FEATURE_THRESHOLD
        agent.perceiver.PLASTICITY_UPDATE_RATE = agent.perceiver.NEW_FEATURE_THRESHOLD * 0.003
        '''
        pass
            
         
    def display(self):
        """ Provide an intuitive display of the current state of the World 
        to the user.
        """        
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
        world_utils.vizualize_pixel_array_feature_set(feature_set, world_name='image_1D', 
                                                      save_eps=True, save_jpg=True)
    