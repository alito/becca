import matplotlib.pyplot as plt
import numpy as np

from worlds.base_world import World as BaseWorld
import worlds.world_tools as wtools

class World(BaseWorld):
    """ 
    One-dimensional visual servo task

    In this task, BECCA can direct its gaze left and right 
    along a mural. It is rewarded for directing it near the center. 
    The mural is not represented using basic features, but rather 
    using raw inputs, which BECCA must build into features. 
    See Chapter 4 of the Users Guide for details.
    Optimal performance is a reward of somewhere around 90 per time step.
    """
    def __init__(self, lifespan=None):
        """ Set up the world """
        BaseWorld.__init__(self, lifespan)
        self.VISUALIZE_PERIOD = 10 ** 3
        self.print_feature_set = True
        self.REWARD_MAGNITUDE = 100.
        self.JUMP_FRACTION = 0.1
        self.STEP_COST = 0.1 * self.REWARD_MAGNITUDE
        self.animate = False 
        self.graphing = True
        self.name_long = 'one dimensional visual world'
        self.name = 'image_1D'
        print "Entering", self.name_long
        self.step_counter = 0
        self.fov_span = 5 
        self.num_sensors = 2 * self.fov_span ** 2
        self.num_actions = 9

        # Initialize the image to be used as the environment
        self.block_image_filename = "./images/bar_test.png" 
        self.data = plt.imread(self.block_image_filename)
        # Convert it to grayscale if it's in color
        if self.data.shape[2] == 3:
            # Collapse the three RGB matrices into one b/w value matrix
            self.data = np.sum(self.data, axis=2) / 3.0
        # Define the size of the field of view, its range of 
        # allowable positions, and its initial position.
        image_width = self.data.shape[1]
        self.MAX_STEP_SIZE = image_width / 2
        self.TARGET_COLUMN = image_width / 2
        self.REWARD_REGION_WIDTH = image_width / 8
        self.NOISE_MAGNITUDE = 0.1
    
        self.column_history = []
        self.fov_height = np.min(self.data.shape)
        self.fov_width = self.fov_height
        self.column_min = np.ceil(self.fov_width / 2)
        self.column_max = np.floor(self.data.shape[1] - self.column_min)
        self.column_position = np.random.random_integers(self.column_min, 
                                                         self.column_max)
        self.block_width = self.fov_width / (self.fov_span + 2)
        self.sensors = np.zeros(self.num_sensors)

    def step(self, action): 
        """ Take one step through the world """
        self.timestep += 1
        self.action = action.ravel() 
        # Actions 0-3 move the field of view to a higher-numbered row 
        # (downward in the image) with varying magnitudes, and 
        # actions 4-7 do the opposite.
        column_step = np.round(self.action[0] * self.MAX_STEP_SIZE / 2 + 
                               self.action[1] * self.MAX_STEP_SIZE / 4 + 
                               self.action[2] * self.MAX_STEP_SIZE / 8 + 
                               self.action[3] * self.MAX_STEP_SIZE / 16 - 
                               self.action[4] * self.MAX_STEP_SIZE / 2 - 
                               self.action[5] * self.MAX_STEP_SIZE / 4 - 
                               self.action[6] * self.MAX_STEP_SIZE / 8 - 
                               self.action[7] * self.MAX_STEP_SIZE / 16)
        column_step = np.round(column_step * (
                1 + self.NOISE_MAGNITUDE * np.random.random_sample() * 2.0 - 
                self.NOISE_MAGNITUDE * np.random.random_sample() * 2.0))
        self.column_position = self.column_position + int(column_step)
        self.column_position = max(self.column_position, self.column_min)
        self.column_position = min(self.column_position, self.column_max)
        # At random intervals, jump to a random position in the world
        if np.random.random_sample() < self.JUMP_FRACTION:
            self.column_position = np.random.random_integers(self.column_min, 
                                                             self.column_max)
        # Create the sensory input vector
        fov = self.data[:, self.column_position - self.fov_width / 2: 
                           self.column_position + self.fov_width / 2]
        center_surround_pixels = wtools.center_surround(fov, 
                                                             self.fov_span)
        unsplit_sensors = center_surround_pixels.ravel()        
        self.sensors = np.concatenate((np.maximum(unsplit_sensors, 0), 
                                  np.abs(np.minimum(unsplit_sensors, 0)) ))
        # Calculate the reward
        self.reward = 0
        if (np.abs(self.column_position - self.TARGET_COLUMN) < 
                self.REWARD_REGION_WIDTH / 2.0):
            self.reward += self.REWARD_MAGNITUDE
        self.reward -= np.abs(column_step) / self.MAX_STEP_SIZE * self.STEP_COST
        return self.sensors, self.reward
            
    def set_agent_parameters(self, agent):
        """ Initalize some of BECCA's parameters to ensure smooth running """
        agent.reward_min = 0.
        agent.reward_max = 100.
        return   

    def visualize(self, agent):
        """ Keep track of what's going on in the world and display it """
        if self.animate:
            print ''.join(['column_position: ', str(self.column_position), 
                           '  self.reward: ', str(self.reward)])
        if not self.graphing:
            return

        # Periodically show the state history and inputs as perceived by BECCA
        self.column_history.append(self.column_position)
        if (self.timestep % self.VISUALIZE_PERIOD) == 0:
            # Periodically show the agent's internal state and reward history
            agent.visualize() 

            print ''.join(["world is ", str(self.timestep), " timesteps old"])
            fig = plt.figure(11)
            plt.clf()
            plt.plot( self.column_history, 'k.')    
            plt.title("Column history")
            plt.xlabel('time step')
            plt.ylabel('position (pixels)')
            fig.show()
            fig.canvas.draw()

            fig  = plt.figure(12)
            sensed_image = np.reshape(
                    0.5 * (self.sensors[:len(self.sensors)/2] - 
                           self.sensors[len(self.sensors)/2:] + 1), 
                    (self.fov_span, self.fov_span))
            plt.gray()
            plt.imshow(sensed_image, interpolation='nearest')
            plt.title("Image sensed")
            fig.show()
            fig.canvas.draw()
            # Periodically visualize the entire feature set
            if self.print_feature_set:
                (feature_set, feature_activities) = agent.get_projections()
                wtools.print_pixel_array_features(feature_set, self.num_sensors, 
                                                  directory='log', world_name=self.name)
        return
