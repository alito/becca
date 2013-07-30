import numpy as np

from cog import Cog
import tools
from ziptie import ZipTie

class Block(object):
    """
    The building block of which the agent is composed

    Blocks are arranged hierarchically within the agent. 
    The agent begins with only one block, and creates additional
    blocks in a tower arrangement as lower ones mature. 
    
    The block's input channels (cables) are organized 
    into clusters (bundles)
    whose activities are passed up to the next block in the hierarchy.  
    Each block performs the same two functions, 
    1) a step_up 
    where cable activities are converted to bundle activities and
    passed up the tower and 
    2) a step_down where bundle activity goals are passed back down
    and converted into cable activity goals. 
    Internally, a block contains a number of cogs that work in parallel
    to convert cable activities into bundle activities and back again.
    """
    def __init__(self, max_cables=240, max_cogs=120,
                 max_cables_per_cog=10, max_bundles_per_cog=2, 
                 name='anonymous', level=0):
        """ Initialize the level, defining the dimensions of its cogs """
        self.max_cables_per_cog = max_cables_per_cog
        self.max_bundles_per_cog = max_bundles_per_cog
        self.max_cogs = max_cogs
        self.max_cables = max_cables
        self.max_bundles = self.max_cogs * self.max_bundles_per_cog
        self.name = name
        self.level = level
        ziptie_name = ''.join(('ziptie_', self.name))
        self.ziptie = ZipTie(self.max_cables, self.max_cogs, 
                             max_cables_per_bundle=self.max_cables_per_cog,
                             mean_exponent=-2,
                             joining_threshold=0.02, name=ziptie_name)
        self.cogs = []
        # TODO: only create cogs as needed
        for cog_index in range(max_cogs):
            self.cogs.append(Cog(self.max_cables_per_cog, 
                                 self.max_bundles_per_cog,
                                 max_chains_per_bundle=self.max_cables_per_cog,
                                 name='cog'+str(cog_index), 
                                 level=self.level))
        self.cable_activities = np.zeros((self.max_cables, 1))
        self.ACTIVITY_DECAY_RATE = .5 # real, 0 < x < 1
        # Constants for adaptively rescaling the cable activities
        self.max_vals = np.zeros((self.max_cables, 1)) 
        self.min_vals = np.zeros((self.max_cables, 1))
        self.RANGE_DECAY_RATE = 10 ** -5
        
    def step_up(self, new_cable_activities, reward):
        """ Find bundle_activities that result from new_cable_activities """
        new_cable_activities = tools.pad(new_cable_activities, 
                                         (self.max_cables, 1))
        # Condition the new_cable_activities to fall between 0 and 1
        self.min_vals = np.minimum(new_cable_activities, self.min_vals)
        self.max_vals = np.maximum(new_cable_activities, self.max_vals)
        spread = self.max_vals - self.min_vals
        new_cable_activities = ((new_cable_activities - self.min_vals) / 
                            (self.max_vals - self.min_vals + tools.EPSILON))
        self.min_vals += spread * self.RANGE_DECAY_RATE
        self.max_vals -= spread * self.RANGE_DECAY_RATE
        # Update cable_activities, incorporating sensing dynamics
        self.cable_activities = tools.bounded_sum([
                new_cable_activities, 
                self.cable_activities * (1. - self.ACTIVITY_DECAY_RATE)])
        # debug 
        #print self.name, 'ca', self.cable_activities.shape
        #print self.cable_activities.ravel()

        # Update the map from self.cable_activities to cogs
        self.ziptie.update(self.cable_activities)
        # Process the upward pass of each of the cogs in the block
        self.bundle_activities = np.zeros((0, 1))
        for cog_index in range(len(self.cogs)):
            # Pick out the cog's cable_activities, process them, 
            # and assign the results to block's bundle_activities
            cog_cable_activities = self.cable_activities[
                    self.ziptie.get_projection(cog_index).ravel().astype(bool)]
            cog_bundle_activities = self.cogs[cog_index].step_up(
                    cog_cable_activities, reward)
            self.bundle_activities = np.concatenate((self.bundle_activities, 
                                                 cog_bundle_activities))
        return self.bundle_activities

    def step_down(self, bundle_activity_goals):
        """ Find cable_activity_goals, given a set of bundle_activity_goals """
        bundle_activity_goals = tools.pad(bundle_activity_goals, 
                                          (self.max_bundles, 1))
        instant_cable_activity_goals = np.zeros((self.max_cables, 1))
        #self.cable_activity_goals = np.zeros((self.max_cables, 1))
        #self.reaction = np.zeros((self.max_cables, 1))
        self.surprise = np.zeros((self.max_cables, 1))
        # Process the downward pass of each of the cogs in the level
        cog_index = 0
        for cog in self.cogs:
            # Gather the goal inputs for each cog
            cog_bundle_activity_goals = bundle_activity_goals[
                    cog_index * self.max_bundles_per_cog:
                    cog_index + 1 * self.max_bundles_per_cog,:]
            # Update the downward outputs for the level 
            instant_cable_activity_goals_by_cog = cog.step_down(
                    cog_bundle_activity_goals)
            cog_cable_indices = self.ziptie.get_projection(
                    cog_index).ravel().astype(bool)
            #instant_cable_activity_goals[cog_cable_indices] = np.maximum(
            #        tools.pad(instant_cable_activity_goals_by_cog, 
            #                  (cog_cable_indices[0].size, 0)),
            #        instant_cable_activity_goals[cog_cable_indices]) 
            instant_cable_activity_goals[cog_cable_indices] = np.maximum(
                    instant_cable_activity_goals_by_cog, 
                    instant_cable_activity_goals[cog_cable_indices]) 
            #self.cable_activity_goals[cog_cable_indices] = np.maximum(
            #        tools.pad(cog.goal_output, (cog_cable_indices[0].size, 0)),
            #        self.cable_activity_goals[cog_cable_indices]) 
            #self.reaction[cog_cable_indices] = np.maximum(
            #        tools.pad(cog.reaction, (cog_cable_indices[0].size, 0)),
            #        self.reaction[cog_cable_indices]) 
            self.surprise[cog_cable_indices] = np.maximum(
                    cog.surprise, self.surprise[cog_cable_indices]) 
            cog_index += 1
        return instant_cable_activity_goals 

    def get_projection(self, bundle_index):
        """ Represent one of the bundles in terms of its cables """
        # Find which cog it belongs to and which output it corresponds to
        cog_index = int(bundle_index / self.max_bundles_per_cog)
        cog_bundle_index = bundle_index - cog_index * self.max_bundles_per_cog
        # Find the projection to the cog's own cables
        cog_cable_indices = self.ziptie.get_projection(
                cog_index).ravel().astype(bool)
        num_cables_in_cog = np.sum(cog_cable_indices)
        cog_projection = self.cogs[cog_index].get_projection(cog_bundle_index)
        # Then re-sort them to the block's cables
        projection = np.zeros((self.max_cables, 2))
        projection[cog_cable_indices,:] = cog_projection[:num_cables_in_cog,:]
        return projection

    def visualize(self):
        """ Show what's going on inside the level """
        self.ziptie.visualize()
        #for cog in self.cogs:
        #    cog.visualize()
        return
