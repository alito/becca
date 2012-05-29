
""" The Python Image Library, required by this world, installed
as part of pyplot. This allows the loading and interpreting of .jpgs
"""
import matplotlib.pyplot as plt
import numpy as np

import agent.viz_utils as viz_utils

#from utils import force_redraw
from worlds.base_world import World as BaseWorld

class World(BaseWorld):
    """ Image_2D
    two-dimensional visual servo task

    In this task, BECCA can direct its gaze up, down, left, and
    right, saccading about an image of a black square on a white
    background. It is rewarded for directing it near the center.
    The mural is not represented using basic features, but rather
    using raw inputs, which BECCA must build into features. See
    
    http://www.sandia.gov/~brrohre/doc/Rohrer11DevelopmentalAgentLearning.pdf
    
    for a full writeup.
    
    Optimal performance is around 0.7 reward per time step.
    """
    def __init__(self):
        super(World, self).__init__()

        self.REPORTING_PERIOD = 10 ** 2       
        self.BACKUP_PERIOD = 10 ** 4
        self.LIFESPAN = 10 ** 5
        self.REWARD_MAGNITUDE = 1.0
        self.ANIMATE_PERIOD = 10 ** 2
        self.animate = True
        self.graphing = True
        
        self.step_counter = 0

        self.fov_span = 5

        self.num_sensors = 2 * self.fov_span ** 2
        self.num_primitives = 1
        self.num_actions = 16

        self.column_history = []
        self.row_history = []

        """ Initialize the image to be used as the environment """
        self.image_filename = "./images/block_test.jpg" 
        self.data = plt.imread(self.image_filename)
        
        """ Convert it to grayscale if it's in color """
        if self.data.shape[2] == 3:
            """ Collapse the three RGB matrices into one black/white value
            matrix.
            """
            self.data = np.sum(self.data, axis=2) / 3.0

        """ Define the size of the field of view, 
        its range of allowable positions,
        and its initial position.
        """
        self.MAX_STEP_SIZE = self.data.shape[1] / 2
        self.TARGET_COLUMN = self.MAX_STEP_SIZE
        self.TARGET_ROW = self.MAX_STEP_SIZE
        self.REWARD_REGION_WIDTH = self.MAX_STEP_SIZE / 2
        self.NOISE_MAGNITUDE = self.MAX_STEP_SIZE * 0.1

        self.FIELD_OF_VIEW_FRACTION = 0.5;
        self.fov_height = np.minimum(self.data.shape[0], self.data.shape[1]) *\
                            self.FIELD_OF_VIEW_FRACTION
        self.fov_width = self.fov_height
        self.column_min = np.ceil(self.fov_width / 2 + 1)
        self.column_max = self.data.shape[1] - self.column_min
        self.row_min = np.ceil(self.fov_height / 2 + 1)
        self.row_max = self.data.shape[1] - self.column_min
        self.column_position = np.random.random_integers(self.column_min, self.column_max)
        self.row_position = np.random.random_integers(self.row_min, self.row_max)

        self.block_width = self.fov_width / self.fov_span
        self.block_height = self.fov_height / self.fov_span

        self.sensors = np.zeros(self.num_sensors)
        self.primitives = np.zeros(self.num_primitives)
        

    def step(self, action): 
        """ Advance the World by one time step """
        self.timestep += 1
        
        """ Actions 0-3 move the field of view to a higher-numbered 
        row (downward in the image) with varying magnitudes, and
        actions 4-7 do the opposite.
        Actions 8-11 move the field of view to a higher-numbered 
        column (rightward in the image) with varying magnitudes, and
        actions 12-15 do the opposite.
        """
        row_step    = np.round(action[0] * self.MAX_STEP_SIZE / 2 + 
                               action[1] * self.MAX_STEP_SIZE / 4 + 
                               action[2] * self.MAX_STEP_SIZE / 8 + 
                               action[3] * self.MAX_STEP_SIZE / 16 - 
                               action[4] * self.MAX_STEP_SIZE / 2 - 
                               action[5] * self.MAX_STEP_SIZE / 4 - 
                               action[6] * self.MAX_STEP_SIZE / 8 - 
                               action[7] * self.MAX_STEP_SIZE / 16)
        column_step = np.round(action[8] * self.MAX_STEP_SIZE / 2 + 
                               action[9] * self.MAX_STEP_SIZE / 4 + 
                               action[10] * self.MAX_STEP_SIZE / 8 + 
                               action[11] * self.MAX_STEP_SIZE / 16 - 
                               action[12] * self.MAX_STEP_SIZE / 2 - 
                               action[13] * self.MAX_STEP_SIZE / 4 - 
                               action[14] * self.MAX_STEP_SIZE / 8 - 
                               action[15] * self.MAX_STEP_SIZE / 16)
        
        row_step    = np.round( column_step * ( 1 + self.NOISE_MAGNITUDE * 
                                np.random.random_sample() - 
                                self.NOISE_MAGNITUDE * 
                                np.random.random_sample()))
        column_step = np.round( column_step * ( 1 + self.NOISE_MAGNITUDE * 
                                np.random.random_sample() - 
                                self.NOISE_MAGNITUDE * 
                                np.random.random_sample()))
        self.row_position = self.row_position + int(row_step)
        self.column_position = self.column_position + int(column_step)

        self.row_position = max(self.row_position, self.row_min)
        self.row_position = min(self.row_position, self.row_max)
        self.column_position = max(self.column_position, self.column_min)
        self.column_position = min(self.column_position, self.column_max)

        """ Create the sensory input vector """
        fov = self.data[self.row_position - self.fov_width / 2: 
                        self.row_position + self.fov_width / 2, 
                        self.column_position - self.fov_width / 2: 
                        self.column_position + self.fov_width / 2]

        sensors = np.zeros(self.num_sensors / 2)

        for row in range(self.fov_span):
            for column in range(self.fov_span):

                sensors[row + self.fov_span * column] = \
                    np.mean( fov[row * self.block_height: (row + 1) * \
                                 self.block_height, 
                                 column * self.block_width: (column + 1) * \
                                 self.block_width ]) / 255.0

        sensors = sensors.ravel()
        sensors = np.concatenate((sensors, 1 - sensors))

        reward = self.calculate_reward()               
        
        self.log(sensors, self.primitives, reward)
        
        return sensors, self.primitives, reward
    
    
    def calculate_reward(self):
        
        reward = 0
        if (abs(self.column_position - self.TARGET_COLUMN) < 
            self.REWARD_REGION_WIDTH / 2) \
            & (abs(self.row_position - self.TARGET_ROW) < 
               self.REWARD_REGION_WIDTH / 2):
            reward = self.REWARD_MAGNITUDE

        return reward

        
    def log(self, sensors, primitives, reward):
        
        self.display()

        self.row_history.append(self.row_position)
        self.column_history.append(self.column_position)

        if self.animate and (self.timestep % self.ANIMATE_PERIOD) == 0:
            plt.figure("Image sensed")
            sensed_image = np.reshape(sensors[:len(sensors)/2], 
                                      (self.fov_span, self.fov_span))
            plt.gray()
            plt.imshow(sensed_image)
            viz_utils.force_redraw()

         
    def display(self):
        """ Provide an intuitive display of the current state of the World 
        to the user.
        """        
        if (self.timestep % self.REPORTING_PERIOD) == 0:
            
            print("world is %s timesteps old" % self.timestep)
            
            if self.graphing:
                plt.figure("Row history")
                plt.clf()
                plt.plot( self.row_history, 'k.')    
                plt.xlabel('time step')
                plt.ylabel('position (pixels)')
                viz_utils.force_redraw()

                plt.figure("Column history")
                plt.clf()
                plt.plot( self.column_history, 'k.')    
                plt.xlabel('time step')
                plt.ylabel('position (pixels)')
                viz_utils.force_redraw()
                            
            return