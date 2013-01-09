
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
    """ watch, visual feature extraction
    In this world, Becca's feature extractor creates visual    
    features from portions of images in the
    Caltech-256 block_image_data dataset. The reinforcement actor serves no
    vital purpose in this world--it is intended to showcase the
    feature extractor.

    By default, this world pulls images from the collection at 
    ./images/lib . This directory is specifically excluded from 
    the repository since it can be quite large and shouldn't be 
    distributed with Becca. You will have to fill this directory
    yourself in order to run this world. See

    http://www.sandia.gov/rohrer/doc/Rohrer11BiologicallyInspiredFeature.pdf

    for a full writeup. 
    
    This world also requires an installation of the Python Imaging Library
    (PIL). I've had problems installing it on Windows. I had to recompile it
    to run it on Mac. It ran on Ubuntu Linux 12.04 right out of the box.
    
    For fastest running of this task, comment out the line that calls the 
    actor in the step() method of agent.py.
    """
    def __init__(self):
        super(World, self).__init__()

        self.TASK_DURATION = 10 ** 1
        self.FEATURE_DISPLAY_INTERVAL = 10 ** 3
        self.LIFESPAN = 10 ** 8
        self.FOV_FRACTION = 0.2
        self.name = 'watch world'
        self.name_short = 'watch'
        self.announce()

        self.timestep = 0
        self.sample_counter = 0

        self.fov_span = 7
        
        self.num_sensors = 2 * self.fov_span ** 2
        self.num_primitives = 0
        self.num_actions = 16

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
            print self.image_count, ' block_image_data filenames loaded.'
            
        """ Initialize the block_image_data to be viewed """
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
            self.block_image_data = np.asarray(self.image) / 255.0    
                    
        else:
            self.block_image_data = plt.imread(filename)
            """ Convert it to grayscale if it's in color """
            if len(self.block_image_data.shape) == 3:
                self.block_image_data = np.sum(self.block_image_data, axis=2) / \
                                    self.block_image_data.shape[2]
            
        (im_height, im_width) = self.block_image_data.shape
        im_size = np.minimum(im_height, im_width)
        
        self.fov_height = im_size * self.FOV_FRACTION
        self.fov_width = self.fov_height

        self.column_min = int(np.ceil( self.fov_width/2)) + 1
        self.column_max = im_width - int(np.ceil( self.fov_width/2)) - 1
        self.row_min = int(np.ceil( self.fov_height/2)) + 1
        self.row_max = im_height - int(np.ceil( self.fov_height/2)) - 1

        self.block_width = int(np.floor(self.fov_width/ (self.fov_span + 2)))
        self.block_height = int(np.floor(self.fov_height/ (self.fov_span + 2)))
        self.MAX_STEP_SIZE = self.fov_height

        if (( self.block_width < 1) | ( self.block_height < 1)):
            self.initialize_image()
        else:
            self.column_position = np.ceil(np.random.random_sample(1) * \
                (self.column_max - self.column_min)) + self.column_min
            self.row_position = np.ceil(np.random.random_sample(1) * \
                (self.row_max - self.row_min)) + self.row_min

        return    


    def step(self, action): 
        self.timestep += 1
        self.sample_counter += 1

        """ Restart the task when appropriate """
        if self.sample_counter >= self.TASK_DURATION:
            self.initialize_image()

        """ Actions 0-3 move the field of view to a higher-numbered 
        row (downward in the block_image_data) with varying magnitudes, and actions 4-7 do the opposite.
        Actions 8-11 move the field of view to a higher-numbered 
        column (rightward in the block_image_data) with varying magnitudes, and actions 12-15 do the opposite.
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
        
        self.row_position = self.row_position + int(row_step)
        self.column_position = self.column_position + int(column_step)

        self.row_position = np.maximum(self.row_position, self.row_min)
        self.row_position = np.minimum(self.row_position, self.row_max)
        self.column_position = np.maximum(self.column_position, self.column_min)
        self.column_position = np.minimum(self.column_position, self.column_max)

        """ Create the sensory input vector """
        fov = self.block_image_data[int(self.row_position - self.fov_height / 2): 
                        int(self.row_position + self.fov_height / 2), 
                        int(self.column_position - self.fov_width / 2): 
                        int(self.column_position + self.fov_width / 2)]
        
        center_surround_pixels = world_utils.center_surround( \
                        fov, self.fov_span, self.block_width, self.block_width, verbose=False)

        unsplit_sensors = center_surround_pixels.ravel()        
        sensors = np.concatenate((np.maximum(unsplit_sensors, 0), \
                                  np.abs(np.minimum(unsplit_sensors, 0)) ))

        reward = 0              
        
        return sensors, self.primitives, reward
    
            
    def set_agent_parameters(self, agent):

        """ Explore on every time step """
        agent.actor.planner.EXPLORATION_FRACTION = 1.0
        
        """ Don't create a model """
        agent.actor.model.MAX_TRANSITIONS = 10 ** 3
        agent.actor.model.SIMILARITY_THRESHOLD = 0.

        agent.perceiver.NEW_FEATURE_THRESHOLD = 0.02
        agent.perceiver.MIN_SIG_COACTIVITY =  0.8 * agent.perceiver.NEW_FEATURE_THRESHOLD
        agent.perceiver.PLASTICITY_UPDATE_RATE = 0.01 * agent.perceiver.NEW_FEATURE_THRESHOLD
        agent.perceiver.DISSIPATION_FACTOR = - 0.5 * np.log2(agent.perceiver.NEW_FEATURE_THRESHOLD)


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
        