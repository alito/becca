
import agent.viz_utils as viz_utils
from worlds.base_world import World as BaseWorld
import worlds.world_utils as world_utils

import matplotlib.pyplot as plt
import numpy as np
import os

""" Import the Python Imaging Library if it can be found.
If not, carry on.
"""
try:
    from PIL import Image
    using_pil = True
except ImportError:
    using_pil = False
    print "PIL (the Python Imaging Library) was not found."
    print "This means that the watch world will only be able to load .png files."
    print "If you want to load .jpgs and other formats, install PIL."


class World(BaseWorld):
    """ Image_2D, two-dimensional visual servo task
    In this task, BECCA can direct its gaze up, down, left, and
    right, saccading about an block_image_data of a black square on a white
    background. It is rewarded for directing it near the center.
    The mural is not represented using basic features, but rather
    using raw inputs, which BECCA must build into features. See
    http://www.sandia.gov/rohrer/doc/Rohrer11DevelopmentalAgentLearning.pdf
    for a full writeup.

    By default, this world pulls images from the collection at 
    ./images/lib . This directory is specifically excluded from 
    the repository since it can be quite large and shouldn't be 
    distributed with Becca. You will have to fill this directory
    yourself in order to run this world. See

    This world also requires an installation of the Python Imaging Library
    (PIL). I've had problems installing it on Windows. I had to recompile it
    to run it on Mac. It ran on Ubuntu Linux 12.04 right out of the box.    
    """
    def __init__(self):
        super(World, self).__init__()

        self.TASK_DURATION = 10 ** 3
        self.TARGET_SIZE_FRACTION = 0.8
        self.REPORTING_PERIOD = 10 ** 3   
        self.FEATURE_DISPLAY_INTERVAL = 10 ** 6
        self.LIFESPAN = 2 * 10 ** 6
        self.REWARD_MAGNITUDE = 100.
        self.ANIMATE_PERIOD = 10 ** 2
        self.animate = False
        self.graphing = False
        self.name = 'find block world'
        self.name_short = 'block'
        self.announce()

        self.sample_counter = 0
        self.step_counter = 0
        self.fov_span = 7

        self.num_sensors = 2 * self.fov_span ** 2
        self.num_primitives = 0
        self.num_actions = 17

        self.column_history = []
        self.row_history = []

        """ Initialize the block_image_data to be used as the environment """
        self.block_image_filename = "./images/block_test.png" 
        self.block_image_data = plt.imread(self.block_image_filename)
        
        """ Convert it to grayscale if it's in color """
        if self.block_image_data.shape[2] == 3:
            """ Collapse the three RGB matrices into one black/white value matrix """
            self.block_image_data = np.sum(self.block_image_data, axis=2) / 3.0
        
        self.image_filenames = []
        path = 'images/lib/' 
        
        if using_pil:
            extensions = ['.jpg', '.tif', '.gif', '.png', '.bmp']
        else:
            extensions = ['.png']

        for localpath, directories, filenames in os.walk(path):
            for filename in filenames:
                for extension in extensions:
                    if filename.endswith(extension):
                        self.image_filenames.append(os.path.join(localpath,filename))
                                                     
        self.image_count = len(self.image_filenames)
        if self.image_count == 0:
            try:
                raise RuntimeError('Add image files to image\/lib\/')
            except RuntimeError:
                print 'Error in watch.py: No images loaded.'
                print '    Make sure the \'images\' directory contains '
                print '    a \'lib\' directory and that it contains'
                print '    some image files.'
                raise
        else:
            print self.image_count, 'image_data filenames loaded.'
            
        """ Initialize the image_data to be viewed """
        self.initialize_image()
        
        self.sensors = np.zeros(self.num_sensors)
        self.primitives = np.zeros(self.num_primitives)
        
        self.last_feature_vizualized = 0


    def initialize_image(self):
        
        self.sample_counter = 0
        filename = self.image_filenames[np.random.randint(0, self.image_count)]
        
        if using_pil:
            self.image = Image.open(filename)
            """ Convert it to grayscale if it's in color """
            self.image = self.image.convert('L')
            self.image_data = np.asarray(self.image) / 255.0    
                    
        else:
            self.image_data = plt.imread(filename)
            """ Convert it to grayscale if it's in color """
            if len(self.image_data.shape) == 3:
                self.image_data = np.sum(self.image_data, axis=2) / \
                                    self.image_data.shape[2]
            
        """ Define the size of the base image """
        (im_height, im_width) = self.image_data.shape
        im_size = np.minimum(im_height, im_width)
        self.MAX_STEP_SIZE = im_size / 2
        self.REWARD_REGION_WIDTH = im_size / 8
        self.NOISE_MAGNITUDE = 0.1
        self.FIELD_OF_VIEW_FRACTION = 0.5
        self.fov_height =  im_size * self.FIELD_OF_VIEW_FRACTION
        self.fov_width = self.fov_height
        self.column_min = np.ceil(self.fov_width / 2)
        self.column_max = np.floor(im_width - self.column_min)
        self.row_min = np.ceil(self.fov_height / 2)
        self.row_max = np.floor(im_height - self.row_min)
        self.column_position = np.random.random_integers(self.column_min, self.column_max)
        self.row_position = np.random.random_integers(self.row_min, self.row_max)
        self.superpixel_width = np.round(self.fov_width / (self.fov_span + 2))
        self.superpixel_height = np.round(self.fov_height / (self.fov_span + 2))
        
        """ Define the size of the block image and its initial position """
        max_column_margin = (1 - self.TARGET_SIZE_FRACTION) * im_width
        max_row_margin = (1 - self.TARGET_SIZE_FRACTION) * im_height
        column_margin = np.floor(np.random.random_sample() * max_column_margin)
        row_margin = np.floor(np.random.random_sample() * max_row_margin)
        target_height =  np.round(im_size * self.TARGET_SIZE_FRACTION)
        target_width = target_height
        self.TARGET_COLUMN = column_margin + target_width / 2
        self.TARGET_ROW = row_margin + target_height / 2
        
        scaled_block_data = self.interpolate_nearest_2D(self.block_image_data, target_height, target_width)
        self.image_data[row_margin:row_margin + target_height, 
                        column_margin:column_margin + target_width] = scaled_block_data
        
        if (( self.superpixel_width < 1) | ( self.superpixel_height < 1)):
            self.initialize_image()

        if self.animate:
            plt.figure('image_with_block')
            plt.imshow(self.image_data)
            plt.gray()
            viz_utils.force_redraw()

        return    


    def interpolate_nearest_2D(self, data, rows, cols):
        
        (data_rows, data_cols) = data.shape
        resampled_rows = np.floor(data_rows * np.arange(rows) / rows)
        resampled_cols = np.floor(data_cols * np.arange(cols) / cols)
        
        """ This shaping is necessary for getting the indices to broadcast 
        correctly when slicing the image.
        """
        resampled_rows = resampled_rows[:,np.newaxis]
        resampled_cols = resampled_cols[np.newaxis,:]
        
        return data[resampled_rows.astype(int), resampled_cols.astype(int)]


    def step(self, action): 
        self.timestep += 1
        self.sample_counter += 1
        
        """ Restart the task when appropriate """
        if self.sample_counter >= self.TASK_DURATION:
            self.initialize_image()

        """ Actions 0-3 move the field of view to a higher-numbered 
        row (downward in the image_data) with varying magnitudes, and actions 4-7 do the opposite.
        Actions 8-11 move the field of view to a higher-numbered 
        column (rightward in the image_data) with varying magnitudes, and actions 12-15 do the opposite.
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
        
        row_step    = np.round( row_step * ( 1 + \
                                self.NOISE_MAGNITUDE * np.random.random_sample() * 2.0 - 
                                self.NOISE_MAGNITUDE * np.random.random_sample() * 2.0))
        column_step = np.round( column_step * ( 1 + \
                                self.NOISE_MAGNITUDE * np.random.random_sample() * 2.0 - 
                                self.NOISE_MAGNITUDE * np.random.random_sample() * 2.0))
        self.row_position    = self.row_position    + int(row_step)
        self.column_position = self.column_position + int(column_step)

        """ Respect the boundaries of the image_data """
        self.row_position = max(self.row_position, self.row_min)
        self.row_position = min(self.row_position, self.row_max)
        self.column_position = max(self.column_position, self.column_min)
        self.column_position = min(self.column_position, self.column_max)

        """ Create the sensory input vector """
        fov = self.image_data[self.row_position - self.fov_height / 2: 
                              self.row_position + self.fov_height / 2, 
                              self.column_position - self.fov_width / 2: 
                              self.column_position + self.fov_width / 2]

        center_surround_pixels = world_utils.center_surround( \
                        fov, self.fov_span, self.superpixel_width, self.superpixel_height)

        unsplit_sensors = center_surround_pixels.ravel()        
        sensors = np.concatenate((np.maximum(unsplit_sensors, 0), \
                                  np.abs(np.minimum(unsplit_sensors, 0)) ))
                
        #if not (sensors < 1).all and (sensors > 0).all:
        #    print 'sensors', sensors

        """ Calculate reward """
        target_distance_sq = (self.column_position - self.TARGET_COLUMN) ** 2 +  \
                             (self.row_position - self.TARGET_ROW) ** 2
                           
        reward = self.REWARD_MAGNITUDE * np.exp(- target_distance_sq / 
                                                (0.5 * self.REWARD_REGION_WIDTH ** 2))

        self.log(sensors, self.primitives, reward)
        return sensors, self.primitives, reward
    
    
    def log(self, sensors, primitives, reward):
        
        self.display()
        self.row_history.append(self.row_position)
        self.column_history.append(self.column_position)

        if self.animate and (self.timestep % self.ANIMATE_PERIOD) == 0:
            #print 'sensors', sensors
            plt.figure("Image sensed")
            full_sensors = 0.5 + (sensors[:len(sensors)/2] - sensors[len(sensors)/2:]) / 2
            #print 'full_sensors', full_sensors
            sensed_image = np.reshape(full_sensors,(self.fov_span, self.fov_span))
            plt.gray()
            plt.imshow(sensed_image, interpolation='nearest')
            plt.title('reward: ' + str(reward))
            viz_utils.force_redraw()

 
    def set_agent_parameters(self, agent):
        agent.perceiver.NEW_FEATURE_THRESHOLD = 0.1
        agent.perceiver.MIN_SIG_COACTIVITY =  0.8 * agent.perceiver.NEW_FEATURE_THRESHOLD
        agent.perceiver.PLASTICITY_UPDATE_RATE = 0.01 * agent.perceiver.NEW_FEATURE_THRESHOLD
        agent.perceiver.DISSIPATION_FACTOR = - 0.5 * np.log2(agent.perceiver.NEW_FEATURE_THRESHOLD)

        agent.actor.SALIENCE_WEIGHT = 1.0
        
        #agent.actor.planner.EXPLORATION_FREQUENCY = 0.001
                
        pass
    
        
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
        
    
    def is_time_to_display(self):
        if (self.timestep % self.FEATURE_DISPLAY_INTERVAL == 0):
            return True
        else:
            return False
        
    
    def vizualize_feature_set(self, feature_set):
        """ Provide an intuitive display of the features created by the agent """
        world_utils.vizualize_pixel_array_feature_set(feature_set, 
                                          start=self.last_feature_vizualized, 
                                          world_name=self.name_short, save_eps=True, save_jpg=True)
        self.last_feature_vizualized = feature_set.shape[0]