
from cog import Cog
import utils as ut

import numpy as np

class Level(object):
    def __init__(self, max_inputs_per_cog=20, max_outputs_per_cog=30, name='anonymous'):
        self.cogs = []
        self.name = name
        self.input_map = np.zeros((0,0))
        self.output_map = np.zeros((0,0))
        self.max_inputs_per_cog = max_inputs_per_cog
        self.max_outputs_per_cog = max_outputs_per_cog
        self.num_feature_inputs = 0
        
        
    def step_up(self, feature_inputs, reward):
        if feature_inputs.size > self.num_feature_inputs:
            # Pad the input map
            self.input_map = ut.pad(self.input_map, (-1, 0))
            # Assign input to a cog
            assigned = False
            cog_index = 0
            while not assigned:
                if cog_index < len(self.cogs):
                    if np.random.random_sample() > self.cogs[cog_index].filled():
                        self.input_map[self.num_feature_inputs, cog_index] = 1.
                        assigned = True
                else:
                    # create a new cog
                    self.cogs.append(Cog(self.max_inputs_per_cog, self.max_outputs_per_cog,
                                         name='cog'+str(cog_index)))
                    self.input_map = ut.pad(self.input_map, (0, -1))
                    self.input_map[self.num_feature_inputs, cog_index] = 1.
                    self.output_map = ut.pad(self.output_map, (0, -1))
                    assigned = True
                cog_index += 1
            self.num_feature_inputs += 1
            # Truncate the feature map to only allow one new feature input per time step
            feature_inputs = feature_inputs[:self.num_feature_inputs,:] 
        feature_outputs = np.zeros((self.output_map.shape[0], 1))
        cog_index = 0
        for cog in self.cogs:
            # pick out the cog's feature inputs, process them, and assign the results to feature outputs
            cog_feature_inputs = feature_inputs[np.nonzero(self.input_map[:, cog_index])[0], :]
            cog_feature_outputs = cog.step_up(cog_feature_inputs, reward)
            output_indices = np.nonzero(self.output_map[:, cog_index])[0]
            # Update the output_map to reflect the creation of any features
            if output_indices.size < cog_feature_outputs.size:
                num_new_features = cog_feature_outputs.size - output_indices.size
                self.output_map = ut.pad(self.output_map, (- num_new_features, 0))
                self.output_map[-num_new_features:, cog_index] = 1.
                output_indices = np.nonzero(self.output_map[:, cog_index])[0]
                feature_outputs = np.zeros((self.output_map.shape[0], 1))
            if output_indices.size > 0:
                feature_outputs[output_indices, :] = cog_feature_outputs
            cog_index += 1
            
        if np.random.random_sample() < 0.001:
            print 'num_cogs', len(self.cogs), ' output map shape', self.output_map.shape
        return feature_outputs

    def step_down(self, goal_inputs):
        # Handle an uncommon condition where the number of goal inputs isn't yet as large as 
        # the number of feature outputs
        if goal_inputs.size < self.output_map.shape[0]:
            goal_inputs = ut.pad(goal_inputs, (self.output_map.shape[0], 0))
        goal_outputs = np.zeros((self.input_map.shape[0], 1))
        cog_index = 0
        for cog in self.cogs:
            goal_indices = np.nonzero(self.output_map[:, cog_index])[0]
            if goal_indices.size > 0:
                try:
                    cog_goal_inputs = (self.output_map * goal_inputs)[goal_indices, cog_index][:,np.newaxis]
                except:
                    import traceback, sys
                    print self.name
                    print 'self.output_map', self.output_map
                    print 'goal_inputs', goal_inputs
                    print 'goal_indices', goal_indices
                    print 'cog_index', cog_index 
                    traceback.print_exc(file=sys.stdout)
                    time.sleep(10000)
            else:
                cog_goal_inputs = np.zeros((0,1))
            cog_goal_outputs = cog.step_down(cog_goal_inputs)
            goal_output_indices = np.nonzero(self.input_map[:, cog_index])
            goal_outputs[goal_output_indices] = ut.pad(cog_goal_outputs, (goal_output_indices[0].size, 0))
            cog_index += 1
        return goal_outputs 

    def get_projections(self):
        projections = []
        for cog in self.cogs:
            projections.append(cog.get_projections())
        return projections
        
        
    def display(self):
        for cog in self.cogs:
            cog.display()
        return
