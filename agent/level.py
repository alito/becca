import numpy as np

from cog import Cog
import utils as ut

class Level(object):
    """
    The building block of which the agent is composed

    Levels are named for the fact that they are arranged hierarchically
    within the agent. It begins with level 0, and creates additional
    levels in the heirarchy as lower ones mature. 
    
    Each level performs
    the same functions, a step_up where data is passed up the hierarchy
    and a step_down where goals are passed back down. 
    Internally, a level contains a number of cogs that work in parallel
    to interpret and learn from the data.
    """
    def __init__(self, max_inputs_per_cog=30, max_outputs_per_cog=250, 
                    name='anonymous'):
        """ Initialize the level, defining the dimensions of its cogs """
        self.cogs = []
        self.name = name
        # Determine which of the level's inputs will be routed to which cog
        self.input_map = np.zeros((0,0))
        # Determine how the outputs of each cog will be organized in the 
        # level's set of outputs
        self.output_map = np.zeros((0,0))
        self.max_inputs_per_cog = max_inputs_per_cog
        self.max_outputs_per_cog = max_outputs_per_cog
        self.num_feature_inputs = 0
        # Constants for adaptively rescaling the inputs
        self.max_vals = np.zeros((0,1)) 
        self.min_vals = np.zeros((0,1))
        self.RANGE_DECAY_RATE = 10 ** -3
        
    def step_up(self, feature_inputs, reward):
        """ Distribute feature_inputs to cogs, and pass their outputs up """
        # Assign any new feature_inputs to a cog
        if feature_inputs.size > self.num_feature_inputs:
            # Pad the input map
            self.input_map = ut.pad(self.input_map, (-1, 0))
            self.max_vals = ut.pad(self.max_vals, (-1, 0), -ut.BIG)
            self.min_vals = ut.pad(self.min_vals, (-1, 0), ut.BIG)
            # Assign input to a cog somewhat randomly
            assigned = False
            cog_index = 0
            while not assigned:
                if cog_index < len(self.cogs):
                    if np.random.random_sample() > \
                            self.cogs[cog_index].fraction_filled():
                        #if True:
                        self.input_map[self.num_feature_inputs, cog_index] = 1.
                        assigned = True
                else:
                    # Create a new cog
                    self.cogs.append(Cog(self.max_inputs_per_cog, 
                                         self.max_outputs_per_cog,
                                         name='cog'+str(cog_index)))
                    self.input_map = ut.pad(self.input_map, (0, -1))
                    self.input_map[self.num_feature_inputs, cog_index] = 1.
                    self.output_map = ut.pad(self.output_map, (0, -1))
                    assigned = True
                cog_index += 1
            self.num_feature_inputs += 1
            # Truncate the feature map to only allow one new feature input 
            # per time step
            feature_inputs = feature_inputs[:self.num_feature_inputs,:] 

        # Condition the inputs to fall between 0 and 1
        self.min_vals = np.minimum(feature_inputs, self.min_vals)
        self.max_vals = np.maximum(feature_inputs, self.max_vals)
        spread = self.max_vals - self.min_vals
        inputs = (feature_inputs - self.min_vals) / \
                        (self.max_vals - self.min_vals + ut.EPSILON)
        self.min_vals += spread * self.RANGE_DECAY_RATE
        self.max_vals -= spread * self.RANGE_DECAY_RATE
        
        # Process the upward pass of each of the cogs in the level
        self.feature_outputs = np.zeros((self.output_map.shape[0], 1))
        cog_index = 0
        for cog in self.cogs:
            # Pick out the cog's feature inputs, process them, and assign 
            # the results to feature outputs
            cog_feature_inputs = inputs[np.nonzero(
				        self.input_map[:, cog_index])[0], :]
            cog_feature_outputs = cog.step_up(cog_feature_inputs, reward)
            output_indices = np.nonzero(self.output_map[:, cog_index])[0]
            # Update the output_map to reflect the creation of any features
            if output_indices.size < cog_feature_outputs.size:
                num_new_features = cog_feature_outputs.size - \
                                   output_indices.size
                self.output_map = ut.pad(self.output_map, 
                                        (-num_new_features, 0))
                self.output_map[-num_new_features:, cog_index] = 1.
                output_indices = np.nonzero(self.output_map[:, cog_index])[0]
                self.feature_outputs = ut.pad(self.feature_outputs,
                                        (-num_new_features, 0))
            # Collect the feature_outputs from each cog
            if output_indices.size > 0:
                self.feature_outputs[output_indices, :] = cog_feature_outputs
            cog_index += 1
        return self.feature_outputs

    def step_down(self, goal_inputs):
        """ Distribute goal_inputs to cogs, and pass their outputs down """
        # Handle an uncommon condition where the number of goal inputs 
        # isn't yet as large as the number of feature outputs
        if goal_inputs.size < self.output_map.shape[0]:
            goal_inputs = ut.pad(goal_inputs, (self.output_map.shape[0], 0))
        goal_vote = np.zeros((self.input_map.shape[0], 1))
        self.goal_output = np.zeros((self.input_map.shape[0], 1))
        self.reaction = np.zeros((self.input_map.shape[0], 1))
        self.surprise = np.zeros((self.input_map.shape[0], 1))
        # Process the downward pass of each of the cogs in the level
        cog_index = 0
        for cog in self.cogs:
            # Gather the goal inputs for each cog
            goal_indices = np.nonzero(self.output_map[:, cog_index])[0]
            if goal_indices.size > 0:
                cog_goal_inputs = ((self.output_map * goal_inputs) 
                                   [goal_indices, cog_index][:,np.newaxis])
            else:
                cog_goal_inputs = np.zeros((0,1))
            # Update the downward outputs for the level 
            cog_goal_vote = cog.step_down(cog_goal_inputs)
            output_indices = np.nonzero(self.input_map[:, cog_index])
            goal_vote[output_indices] = np.maximum(
                    ut.pad(cog_goal_vote, (output_indices[0].size, 0)),
                    goal_vote[output_indices]) 
            self.goal_output[output_indices] = np.maximum(
                    ut.pad(cog.goal_output, (output_indices[0].size, 0)),
                    self.goal_output[output_indices]) 
            self.reaction[output_indices] = np.maximum(
                    ut.pad(cog.reaction, (output_indices[0].size, 0)),
                    self.reaction[output_indices]) 
            self.surprise[output_indices] = np.maximum(
                    ut.pad(cog.surprise, (output_indices[0].size, 0)),
                    self.surprise[output_indices]) 
            cog_index += 1
        return goal_vote 

    def get_projection(self, feature_index):
        """ Represent the one of the feature outputs in terms of its inputs """
        # Find which cog it belongs to and which output it corresponds to
        cog_index = np.nonzero(self.output_map[feature_index, :])[0]
        cog_feature_index = np.nonzero(np.nonzero(
                self.output_map[:,cog_index])[0] == feature_index)[0]
        # Find the projection to the cog's own inputs
        cog_projection = self.cogs[cog_index].get_projection(cog_feature_index)
        # Translate the projection back into the level's inputs
        num_cog_inputs = np.nonzero(self.input_map[:,cog_index])[0].size
        cog_input_feature_indices = np.nonzero(self.input_map[:,cog_index])[0]
        projection = np.zeros((self.input_map.shape[0],2))
        projection[cog_input_feature_indices,:] = \
                cog_projection[:num_cog_inputs,:]
        return projection
        
    def display(self):
        """ Show what's going on inside the level """
        for cog in self.cogs:
            cog.display()
        return
