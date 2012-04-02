
""" The Python Image Library, required by this world """
#import PIL.Image as Image
#import Image
import numpy as np

import matplotlib.pyplot as plt

import utils

#from utils import force_redraw
from worlds.base_world import World as BaseWorld

class World(BaseWorld):
    """ Image_1D
    one-dimensional visual servo task

    In this task, BECCA can direct its gaze left and right along a
    mural. It is rewarded for directing it near the center. The
    mural is not represented using basic features, but rather using
    raw inputs, which BECCA must build into features. For a full writeup see:

    http://www.sandia.gov/~brrohre/doc/Rohrer11ImplementedArchitectureFeature.pdf

    Optimal performance is between 0.7 and 0.8 reward per time step.
    """

    MAX_NUM_FEATURES = 500
    Image_Filename = "./images/bar_test.jpg" 
    
    def __init__(self):
        super(World, self).__init__()

        self.REPORTING_PERIOD = 10 ** 2       
        self.BACKUP_PERIOD = 10 ** 3
        self.LIFESPAN = 10 ** 4
        self.AnimatePeriod = 10 ** 2
        self.animate = False
        self.graphing = False
        
        self.step_counter = 0

        self.fov_span = 10

        self.num_sensors = 2 * self.fov_span ** 2
        self.num_primitives = 1
        self.num_actions = 16

        self.column_history = []

        # initializes the image to be used as the environment
        filename = self.Image_Filename

        #image = Image.open(filename)
        self.data = plt.imread(filename)
        
        """ Convert it to grayscale if it's in color """
        if self.data.shape[2] == 3:
            """ Collapse the three RGB matrices into one black/white value
            matrix.
            """
            self.data = np.sum(self.data, axis=2) / 3.0

        self.MAX_STEP_SIZE = self.data.shape[1] / 2
        self.TARGET_COLUMN = self.MAX_STEP_SIZE

        self.fov_height = self.data.shape[0]
        self.fov_width = self.data.shape[0]
        self.column_min = self.fov_width / 2
        self.column_max = self.data.shape[1] - self.column_min
        self.column_position = np.random.random_integers(self.column_min, self.column_max)

        self.block_width = self.fov_width / self.fov_span

        self.sensors = np.zeros(self.num_sensors)
        self.primitives = np.zeros(self.num_primitives)

        '''if self.animate:
            plt.figure("Image sensed")
            plt.gray() # set to grayscale
            plt.draw()
        '''        

    def calculate_reward(self):
        DISTANCE_FACTOR = self.MAX_STEP_SIZE / 16

        reward = 0
        if abs(self.column_position - self.TARGET_COLUMN) < DISTANCE_FACTOR:
            reward = 1

        return reward

        
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
                utils.force_redraw()
                
                
            return
            
    def log(self, sensors, primitives, reward):
        
        self.column_history.append(self.column_position)

        if self.animate and (self.timestep % self.AnimatePeriod) == 0:
            plt.figure("Image sensed")

            """ remaps [0, 1] to [0, 4/5] for display
            sensed_image = sensed_image / 1.25
            """
            sensed_image = np.reshape( sensors[:len(sensors)/2], (self.fov_span, self.fov_span))
            plt.imshow(sensed_image)
            plt.draw()
            utils.force_redraw()

        
    def step(self, action): 
        """ advances the World by one timestep.
        """

        self.timestep += 1
        
        column_step = np.round(action[0] * self.MAX_STEP_SIZE / 2 + 
                               action[1] * self.MAX_STEP_SIZE / 4 + 
                               action[2] * self.MAX_STEP_SIZE / 8 + 
                               action[3] * self.MAX_STEP_SIZE / 16 - 
                               action[4] * self.MAX_STEP_SIZE / 2 - 
                               action[5] * self.MAX_STEP_SIZE / 4 - 
                               action[6] * self.MAX_STEP_SIZE / 8 - 
                               action[7] * self.MAX_STEP_SIZE / 16)
        
        column_step = np.round( column_step * ( 1 + 0.1 * np.random.random_sample() - 0.1 * np.random.random_sample()))
        self.column_position = self.column_position + int(column_step)

        self.column_position = max(self.column_position, self.column_min)
        self.column_position = min(self.column_position, self.column_max)

        # creates sensory input vector
        fov = self.data[:, self.column_position - self.fov_width / 2: self.column_position + self.fov_width / 2]

        sensors = np.zeros(self.num_sensors / 2)

        for row in range(self.fov_span):
            for column in range(self.fov_span):

                sensors[row + self.fov_span * column] = \
                    np.mean( fov[row * self.block_width: (row + 1) * self.block_width, 
                    column * self.block_width: (column + 1) * self.block_width ]) / 256.0

        sensors = sensors.ravel()
        sensors = np.concatenate((sensors, 1 - sensors))

        reward = self.calculate_reward()               
        
        self.log(sensors, self.primitives, reward)
        self.display()
        
        return sensors, self.primitives, reward
        