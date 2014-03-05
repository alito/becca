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
    def __init__(self, min_cables, name='anonymous', level=0):
        """ Initialize the level, defining the dimensions of its cogs """
        self.max_cables = int(2 ** np.ceil(np.log2(min_cables)))
        self.max_cables_per_cog = 8
        self.max_bundles_per_cog = 4
        self.max_cogs = self.max_cables / self.max_bundles_per_cog
        self.max_bundles = self.max_cogs * self.max_bundles_per_cog
        self.name = name
        self.level = level
        ziptie_name = ''.join(('ziptie_', self.name))
        self.ziptie = ZipTie(self.max_cables, self.max_cogs, 
                             max_cables_per_bundle=self.max_cables_per_cog,
                             mean_exponent=-2, name=ziptie_name)
        self.cogs = []
        # TODO: only create cogs as needed
        for cog_index in range(self.max_cogs):
            self.cogs.append(Cog(self.max_cables_per_cog, 
                                 self.max_bundles_per_cog,
                                 max_chains_per_bundle=self.max_cables_per_cog,
                                 name='cog'+str(cog_index), 
                                 level=self.level))
        self.cable_activities = np.zeros((self.max_cables, 1))
        self.hub_cable_goals = np.zeros((self.max_cables, 1))
        self.fill_fraction_threshold = .7
        self.ACTIVITY_DECAY_RATE = 1.# real, 0 < x < 1
        # Constants for adaptively rescaling the cable activities
        self.max_vals = np.zeros((self.max_cables, 1)) 
        self.min_vals = np.zeros((self.max_cables, 1))
        self.RANGE_DECAY_RATE = 10 ** -5
        
    def step_up(self, new_cable_activities):
        """ Find bundle_activities that result from new_cable_activities """
        # Condition the cable activities to fall between 0 and 1
        if new_cable_activities.size < self.max_cables:
            new_cable_activities = tools.pad(new_cable_activities, 
                                             (self.max_cables, 1))
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

        # Update the map from self.cable_activities to cogs
        self.ziptie.step_up(self.cable_activities)
        # Process the upward pass of each of the cogs in the block
        self.bundle_activities = np.zeros((0, 1))
        for cog_index in range(len(self.cogs)):
            # Pick out the cog's cable_activities, process them, 
            # and assign the results to block's bundle_activities
            cog_cable_activities = self.cable_activities[
                    self.ziptie.get_index_projection(
                    cog_index).ravel().astype(bool)]
            # Cogs are only allowed to start forming bundles once 
            # the number of cables exceeds the fill_fraction_threshold
            enough_cables = (self.ziptie.cable_fraction_in_bundle(cog_index)
                             > self.fill_fraction_threshold)
            cog_bundle_activities = self.cogs[cog_index].step_up(
                    cog_cable_activities, enough_cables)
            self.bundle_activities = np.concatenate((self.bundle_activities, 
                                                     cog_bundle_activities))
        # Goal fulfillment and decay
        self.hub_cable_goals -= self.cable_activities
        self.hub_cable_goals *= self.ACTIVITY_DECAY_RATE
        self.hub_cable_goals = np.maximum(self.hub_cable_goals, 0.)
        return self.bundle_activities

    def step_down(self, bundle_goals):
        """ Find cable_activity_goals, given a set of bundle_goals """
        bundle_goals = tools.pad(bundle_goals, (self.max_bundles, 1))
        cable_goals = np.zeros((self.max_cables, 1))
        self.surprise = np.zeros((self.max_cables, 1))
        # Process the downward pass of each of the cogs in the level
        cog_index = 0
        for cog in self.cogs:
            # Gather the goal inputs for each cog
            cog_bundle_goals = bundle_goals[
                    cog_index * self.max_bundles_per_cog:
                    cog_index + 1 * self.max_bundles_per_cog,:]
            # Update the downward outputs for the level 
            cable_goals_by_cog = cog.step_down(cog_bundle_goals)
            cog_cable_indices = self.ziptie.get_index_projection(
                    cog_index).ravel().astype(bool)
            cable_goals[cog_cable_indices] = np.maximum(
                    cable_goals_by_cog, cable_goals[cog_cable_indices]) 
            #self.reaction[cog_cable_indices] = np.maximum(
            #        tools.pad(cog.reaction, (cog_cable_indices[0].size, 0)),
            #        self.reaction[cog_cable_indices]) 
            self.surprise[cog_cable_indices] = np.maximum(
                    cog.surprise, self.surprise[cog_cable_indices]) 
            cog_index += 1
        #self.hub_cable_goals = tools.bounded_sum([self.hub_cable_goals, 
        #                                          cable_goals])
        return self.hub_cable_goals 

    def get_index_projection(self, bundle_index):
        """ Represent one of the bundles in terms of its cables """
        # Find which cog it belongs to and which output it corresponds to
        cog_index = int(bundle_index / self.max_bundles_per_cog)
        cog_bundle_index = bundle_index - cog_index * self.max_bundles_per_cog
        # Find the projection to the cog's own cables
        cog_cable_indices = self.ziptie.get_index_projection(
                cog_index).ravel().astype(bool)
        num_cables_in_cog = np.sum(cog_cable_indices)
        cog_projection = self.cogs[cog_index].get_index_projection(
                cog_bundle_index)
        # Then re-sort them to the block's cables
        projection = np.zeros((self.max_cables, 2))
        projection[cog_cable_indices,:] = cog_projection[:num_cables_in_cog,:]
        return projection

    def bundles_created(self):
        total = 0.
        for cog in self.cogs:
            # Check whether all cogs have created all their bundles
                total += cog.num_bundles()
        if np.random.random_sample() < 0.01:
            print total, 'bundles in', self.name, ', max of', self.max_bundles
        return total

    def visualize(self):
        """ Show what's going on inside the level """
        self.ziptie.visualize()
        #for cog in self.cogs:
        #    cog.visualize()
        return
