
import matplotlib.pyplot as plt
import numpy as np
import os
from worlds.base_world import World as BaseWorld

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
    """ watch
    visual feature creation

    In this world, Becca's feature creator creates visual    
    features from portions of images in the
    Caltech-256 image_data dataset. The reinforcement learner serves no
    vital purpose in this world--it is intended to showcase the
    feature creator.
    
    By default, this world pulls images from the collection at 
    ./images/lib . This directory is specifically excluded from 
    the repository since it can be quite large and shouldn't be 
    distributed with Becca. You will have to fill this directory
    yourself in order to run this world. See

    http://www.sandia.gov/~brrohre/doc/Rohrer11BiologicallyInspiredFeature.pdf

    for a full writeup. 
    
    This world also requires an installation of the Python Imaging Library
    (PIL). I've had problems installing it on Windows. I had to recompile it
    to run it on Mac. It ran on Ubuntu Linux 12.04 right out of the box.
    
    For fastest running of this task, comment out the line that calls the 
    learner in the step() method of agent.py.
    """
    def __init__(self):
        super(World, self).__init__()

        self.TASK_DURATION = 10 ** 2
        self.FEATURE_DISPLAY_INTERVAL = 10 ** 2
        self.LIFESPAN = 10 ** 6
        self.FOV_FRACTION = 0.2
        
        self.timestep = 0
        self.sample_counter = 0

        self.fov_span = 7
        
        self.num_sensors = 2 * self.fov_span ** 2
        self.num_primitives = 1
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
                        self.image_filenames.append(os.path.join
                                                    (localpath,filename))
                                                     
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
            print self.image_count, ' image_data filenames loaded.'
            
        """ Initialize the image_data to be viewed """
        self.initialize_image()
        
        self.sensors = np.zeros(self.num_sensors)
        self.primitives = np.zeros(self.num_primitives)
        
        
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
                self.image_data = np.sum(self.image_data, axis=2) / 3.0
            
        self.fov_height = np.minimum(self.image_data.shape[0], 
                                     self.image_data.shape[1]) * self.FOV_FRACTION
        self.fov_width = self.fov_height

        self.column_min = int(np.ceil( self.fov_width/2)) + 1
        self.column_max = self.image_data.shape[1] - \
                            int(np.ceil( self.fov_width/2)) - 1
        self.row_min = int(np.ceil( self.fov_height/2)) + 1
        self.row_max = self.image_data.shape[0] - \
                            int(np.ceil( self.fov_height/2)) - 1

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
        """ Advance the World by one time step """
        self.timestep += 1
        self.sample_counter += 1

        """ Restart the task when appropriate """
        if self.sample_counter >= self.TASK_DURATION:
            self.initialize_image()

        """ Actions 0-3 move the field of view to a higher-numbered 
        row (downward in the image_data) with varying magnitudes, and
        actions 4-7 do the opposite.
        Actions 8-11 move the field of view to a higher-numbered 
        column (rightward in the image_data) with varying magnitudes, and
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
        
        self.row_position = self.row_position + int(row_step)
        self.column_position = self.column_position + int(column_step)

        self.row_position = np.maximum(self.row_position, self.row_min)
        self.row_position = np.minimum(self.row_position, self.row_max)
        self.column_position = np.maximum(self.column_position, self.column_min)
        self.column_position = np.minimum(self.column_position, self.column_max)

        """ Create the sensory input vector """
        fov = self.image_data[int(self.row_position - self.fov_height / 2): 
                        int(self.row_position + self.fov_height / 2), 
                        int(self.column_position - self.fov_width / 2): 
                        int(self.column_position + self.fov_width / 2)]
        
        sensors = np.zeros(self.num_sensors / 2)

        for row in range(self.fov_span):
            for column in range(self.fov_span):

                sensors[row + self.fov_span * column] = \
                    np.mean( fov[row * self.block_height: (row + 1) * \
                                 self.block_height, 
                                 column * self.block_width: (column + 1) * \
                                 self.block_width ])
        """ TODO: Implement a center-surround filter """
        #self.sensory_input = (1 +  util_sigm( 10 * util_center_surround(...
        #reshape(task.sensory_input,[task.fov_span + 2 task.fov_span + 2]))))/2;

        sensors = sensors.ravel()
        sensors = np.concatenate((sensors, 1 - sensors))

        reward = self.calculate_reward()               
        
        return sensors, self.primitives, reward
    
    
    def calculate_reward(self):
        
        reward = 0
        return reward

        
    def set_agent_parameters(self, agent):

        """ Explore on every time step """
        agent.learner.planner.EXPLORATION_FRACTION = 1.0
        agent.learner.planner.OBSERVATION_FRACTION = 0.0
        
        """ Build more tightly co-active groups """
        agent.grouper.MIN_SIG_COACTIVITY = 0.27
        #agent.grouper.MIN_SIG_COACTIVITY = 0.027
        #agent.grouper.PLASTICITY_UPDATE_RATE = 2 * 10 ** (-2) # debug
        agent.learner.model.MAX_ENTRIES = 10 ** 2
        agent.learner.model.SIMILARITY_THRESHOLD = 0.
        
    
    def is_time_to_display(self):
        if (self.timestep % self.FEATURE_DISPLAY_INTERVAL == 0):
            return True
        else:
            return False
        
    
    def vizualize_feature_set(self, feature_set, save_eps=False, 
                              epsfilename='log/feature_set_watch.eps'):
        """ Provide an intuitive display of the features created by the 
        agent. 
        """
        """ feature_set is a list of lists of State objects """
               
        if len(feature_set) == 0:
            return
        
        """ Calculate the number of pixels that span the field of view """
        n_pixels = feature_set[0][0].sensors.size / 2
        fov_span = np.sqrt(n_pixels)

        """ The number of black pixels surrounding each feature """
        border = 1
        
        """ The number of gray pixels between all features """
        gap = 3
        
        """ The contrast factor. 1 is unchanged. 2 is high contrast. """
        contrast = 1.
        
        """ Find the size of the overall image_data """
        n_groups = len(feature_set)
        n_features_max = 0
        for group_index in range(n_groups):
            if len(feature_set[group_index]) > n_features_max:
                n_features_max = len(feature_set[group_index])

        n_pixel_columns = n_features_max * (gap + 2 * border + fov_span) + gap
        n_pixel_rows = n_groups * (gap + 2 * border + fov_span) + gap
        feature_image = 0.8 * np.ones((n_pixel_rows, n_pixel_columns))
        
        """ Populate each feature in the feature image_data """
        for group_index in range(n_groups):
            for feature_index in range(len(feature_set[group_index])):
                sensors = feature_set[group_index][feature_index].sensors
                sensors = sensors * contrast
                pixel_values = ((sensors[0:n_pixels] - \
                                 sensors[n_pixels:2 * n_pixels]) \
                                 + 1.0) / 2.0
                feature_pixels = pixel_values.reshape(fov_span, fov_span)
                feature_image_first_row = group_index * \
                            (gap + 2 * border + fov_span) + gap + border 
                feature_image_last_row = feature_image_first_row + fov_span
                feature_image_first_column = feature_index * \
                            (gap + 2 * border + fov_span) + gap + border 
                feature_image_last_column = feature_image_first_column + \
                                            fov_span
                feature_image[feature_image_first_row:
                              feature_image_last_row,
                              feature_image_first_column:
                              feature_image_last_column] = feature_pixels
                              
                """ Write North border """
                feature_image[feature_image_first_row - border:
                              feature_image_first_row,
                              feature_image_first_column - border:
                              feature_image_last_column + border] = 0
                """ Write South border """
                feature_image[feature_image_last_row:
                              feature_image_last_row + border,
                              feature_image_first_column - border:
                              feature_image_last_column + border] = 0
                """ Write East border """
                feature_image[feature_image_first_row - border:
                              feature_image_last_row + border,
                              feature_image_first_column - border:
                              feature_image_first_column] = 0
                """ Write West border """
                feature_image[feature_image_first_row - border:
                              feature_image_last_row + border,
                              feature_image_last_column:
                              feature_image_last_column + border] = 0
                
        fig = plt.figure("watch world features")
        plt.gray()
        imgplot = plt.imshow(feature_image, vmin=0.0, vmax=1.0)
        imgplot.set_interpolation('nearest')
        plt.title("Features created while in the watch world")
        plt.draw()
        
        if save_eps:
            fig.savefig(epsfilename, format='eps')
            
        return
        