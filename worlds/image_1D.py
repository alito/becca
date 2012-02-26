import sys
import logging
from collections import defaultdict

import PIL.Image as Image
import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    #no matplotlib, no graphs
    pass


from .world import World

class Image_1D(World):
    ''' Image_1D
    one-dimensional visual servo task

    In this task, BECCA can direct its gaze left and right along a
    mural. It is rewarded for directing it near the center. The
    mural is not represented using basic features, but rather using
    raw inputs, which BECCA must build into features. See

    http://www.sandia.gov/~brrohre/doc/Rohrer11ImplementedArchitectureFeature.pdf

    for a full writeup.

    Simulation only.

    Both MATLAB and Octave compatible.

    Optimal performance is between 0.7 and 0.8 reward per time step.
    '''

    MAX_NUM_FEATURES = 500
    Image_Filename = './images/bar_test.jpg' 
    
    def __init__(self):
        super(Image_1D, self).__init__()

        self.REPORTING_BLOCK_SIZE = 10 ** 2
        self.REPORTING_PERIOD = 10 ** 2        
        self.BACKUP_PERIOD = 10 ** 3
        self.LIFESPAN = 10 ** 4
        self.AnimatePeriod = 10
        
        self.fov_span = 10

        self.num_sensors = 2 * self.fov_span ** 2
        self.num_primitives = 1
        self.num_actions = 16
        

        self.step_counter = 0

        self.animate = True
        self.column_history = []

        # initializes the image to be used as the environment
        filename = self.Image_Filename

        image = Image.open(filename)
        # convert it to grayscale
        if image.mode != 'L':
            grayscale = image.convert('L')
        else:
            grayscale = image

        #load it into a numpy array as doubles
        self.data = np.array(grayscale.getdata()).reshape(grayscale.size[1], grayscale.size[0]).astype('double')
        
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

        if self.graphing:
            if self.animate:
                plt.figure("Image sensed")
                plt.gray() # set to grayscale


    def calculate_reward(self):
        DISTANCE_FACTOR = self.MAX_STEP_SIZE / 16

        reward = 0
        if abs(self.column_position - self.TARGET_COLUMN) < DISTANCE_FACTOR:
            reward = 1

        return reward

    
    
    def display(self):
        ''' provides an intuitive display of the current state of the World 
        to the user
        '''
            
        if (self.timestep % self.REPORTING_PERIOD) == 0:
            
            logging.info("%s timesteps done" % self.timestep)
            
            self.record_reward_history()
            self.cumulative_reward = 0
            self.show_reward_history()

            if self.graphing:
                plt.figure("Column history")
                plt.clf()
                plt.plot( self.column_history, 'k.')    
                plt.xlabel('time step')
                plt.ylabel('position (pixels)')
                # pause is needed for events to be processed
                # Qt backend needs two event rounds to process screen. Any number > 0.01 and <=0.02 would do
                self._force_draw()
                

            
    def log(self, sensors, primitives, reward):
        World.log(self, sensors, primitives, reward)
        
        self.column_history.append(self.column_position)

        if self.animate and (self.timestep % self.AnimatePeriod) == 0:
            plt.figure("Image sensed")
            sensed_image = np.reshape( sensors[:len(sensors)/2], (self.fov_span, self.fov_span))
            #remaps [0, 1] to [0, 4/5] for display
            #sensed_image = sensed_image / 1.25
            plt.imshow(sensed_image)
            self._force_draw()

            """
            figure(7)
            column_vals = 1:size(self.data,1)
            row_vals1 = ones(size(column_vals)) * ...
                (self.column_position - floor( self.fov_width/2))
            row_vals2 = ones(size(column_vals)) * ...
                (self.column_position + ceil ( self.fov_width/2))
            data_temp = self.data
            data_temp(column_vals, row_vals1) = 0
            data_temp(column_vals, row_vals2) = 0
            image(data_temp)
            drawnow
            """

    def show_features(self):
        # initializes feature sets to display
        """
        num_groups_display = length(self.agent.feature_activity)
        span = self.fov_span

        for k = 1:num_groups_display,
            if k == 1
                empty_features{k} = zeros( self.state_length, 1)
            else
                empty_features{k} = zeros( size( self.agent.feature_activity{k}))
            end
        end

        #builds features
        for k = 4:num_groups_display,

            for k2 = 1: size( self.agent.feature_map.map{k},1),
                this_feature = empty_features
                this_feature{k}(k2) = 1
                this_feature_final{k, k2} = agent_expand(self.agent, this_feature)
            end
        end

        pos_index = 1:self.state_length / 2
        neg_index = self.state_length/2 + 1: self.state_length
        pos_index = pos_index(:)
        neg_index = neg_index(:)

        #creates the image set representing all features individually
        feature_image_set = []
        for k = 4:num_groups_display,
            num_features_in_group = size(self.agent.feature_map.map{k},1)
            for k2 = 1:num_features_in_group

               #remaps the interval(-1, 1) to (0, 4/5) for display
                pos_mask = find( this_feature_final{k,k2}{1}(pos_index) < 2)
                neg_mask = find( this_feature_final{k,k2}{1}(neg_index) < 2)        
                mask_index = union( pos_mask, neg_mask)

                this_feature_pixels = ones( size( pos_index))
                this_feature_pixels( pos_mask) = 0
                this_feature_pixels( neg_mask) = 0
                this_feature_pixels( pos_mask) = ...
                    this_feature_pixels( pos_mask) + ...
                    this_feature_final{k,k2}{1}(pos_mask)
                this_feature_pixels( neg_mask) = ...
                    this_feature_pixels( neg_mask) - ...
                    this_feature_final{k,k2}{1}(neg_mask + self.state_length/2)

                this_feature_pixels( mask_index) = ...
                    (util_sigm(this_feature_pixels( mask_index)) + 1) / 2.5 

                feature_image = reshape( this_feature_pixels, span, span)
                feature_image_border = horzcat( zeros( span, 1), ...
                    feature_image, zeros( span, 1))
                feature_image_border = vertcat( zeros( 1, span + 2), ...
                    feature_image_border, zeros( 1, span + 2))

                #adds the extra border indicating activity level
                [f_height f_width] = size(feature_image_border)
                feature_background = ones( f_height+2, f_width+2)

                feature_background(2:end-1,2:end-1) = feature_image_border
                feature_image = feature_background
                feature_image_set{k,k2} = feature_image

            end
        end

        if ~isempty( feature_image_set)

            figure(4)
            clf
            num_feats_in_group = size( feature_image_set,2)
            rows_in_feat = size( feature_image_set{4,1},1)
            cols_in_feat = size( feature_image_set{4,1},2)
            master_feature = ones( num_groups_display * (rows_in_feat + 2), ...
                                   num_feats_in_group * (cols_in_feat + 2))
            for k = 4: num_groups_display,
                for k2 = 1: num_feats_in_group,
                    if ~isempty( feature_image_set{k,k2})
                        mask_index = find( feature_image_set{k,k2} < 1.0)
                        this_feature_pixels = ones( size(feature_image_set{k,k2}))
                        this_feature_pixels( mask_index) = ...
                            feature_image_set{k,k2}( mask_index)
                        grp = k - 3
                        master_feature(( grp - 1) * (rows_in_feat + 2) + 2: ...
                                         grp      * (rows_in_feat + 2) - 1, ...
                                       (k2 - 1) * (cols_in_feat + 2) + 2: ...
                                        k2      * (cols_in_feat + 2) - 1) = ...
                                        this_feature_pixels
                    end
                end
            end
            image(master_feature * 256)
            axis off
            axis equal
            set(4,'Name', 'features by group')

            print( '-f4', '-deps', ['log/log_elements_' datestr(floor(now)) '.eps'])

            figure(1)
            sensory_composite = self.sensory_input( pos_index) - ...
                                self.sensory_input( neg_index)
            feature_image = reshape( sensory_composite, span, span)
            #remaps [-1, 1] to [0, 4/5] for display
            feature_image = (feature_image+1)/2.5
            #remaps [0, 1] to [0, 4/5] for display
            #feature_image = feature_image / 1.25
            image(feature_image * 256)
            title('sensed inputs')
            title('sensed')
            axis equal
        """
        
    def step(self, action): 
        ''' advances the World by one timestep.
        '''

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
        
        
    def final_performance(self):
        ''' When the world terminates, this returns the average performance 
        of the agent on the last 3 blocks.
        '''
        if self.timestep > self.LIFESPAN:
            performance = np.mean(self.reward_history[-3:]) / self.REPORTING_PERIOD
            
            assert(performance >= -1.)
            return performance
        
        return -2
