import numpy as np

import utils as ut

class ZipTie(object):
    """ 
    An incremental unsupervised learning algorithm

    Input channels are clustered togehter into mutually co-active sets.
    A helpful metaphor is bundling cables together with zip ties.
    Cables that carry related signals are commonly co-active, 
    and are grouped together. Cables that are co-active with existing
    bundles can be added to those bundles. A single cable may be ziptied
    into several different bundles. Co-activity is estimated 
    incrementally; the algorithm updates the estimate after each new set 
    of signals is received. 
    """
    def __init__(self, max_num_cables, max_num_bundles, 
                 mean_exponent=-7, name='ziptie_'):
        """ Initialize each map, pre-allocating max_num_bundles """
        self.name = name
        self.max_num_cables = max_num_cables
        self.max_num_bundles = max_num_bundles
        self.num_bundles = 0
        # User-defined constants
        #
        # real, 0 < x < 1, small
        self.COACTIVITY_UPDATE_RATE = 10. ** -2
        # The rate at which the nonbundle_activity thresholds are updated
        # real, 0 < x < 1, small
        self.NEW_BUNDLE_UPDATE_RATE = 10. ** -4
        # Constant factor driving the rate at which new bundles are created
        # real, 0 < x < 1, small
        self.NEW_BUNDLE_FACTOR = 10. ** -2
        # Coactivity value which, if it's every exceeded, causes a 
        # cable to be added to a bundle
        # real, 0 < x < 1, small
        self.JOINING_THRESHOLD = 0.3
        # Exponent for calculating the generalized mean of signals in 
        # order to find bundle activities
        # real, x != 0
        self.MEAN_EXPONENT = mean_exponent
        
        self.bundles_full = False        
        self.bundle_activity = np.zeros((self.max_num_bundles, 1))
        map_size = (self.max_num_bundles, self.max_num_cables)
        self.bundle_map = np.zeros(map_size)
        self.bundle_coactivity = np.zeros(map_size)
        self.cable_coactivity = np.zeros(map_size)
        self.coactivity = np.zeros(map_size)
        self.typical_nonbundle_activity = np.zeros((self.max_num_cables, 1))

    def update(self, new_signals):
        """ Update co-activity estimates and calculate bundle activity """
        # Find bundle activities by taking the generalized mean of
        # the signals with a negative exponent.
        # The negative exponent weights the lower signals more heavily.
        # At the extreme value of negative infinity, the 
        # generalized mean becomes the minimum operator.
        shifted_signals = new_signals + 1.
        signals_to_power = shifted_signals ** self.MEAN_EXPONENT
        mean_signals_to_power = ut.weighted_average(signals_to_power,
                                                    self.bundle_map.T)
        shifted_activity = mean_signals_to_power ** (1./self.MEAN_EXPONENT)
        self.bundle_activity[:self.num_bundles,:] = (
                shifted_activity - 1.)[:self.num_bundles]
        # Calculate how much energy each signal has left to contribute 
        # to the co-activity estimate
        final_activated_bundle_map = self.bundle_activity * self.bundle_map
        combined_weights = np.sum(final_activated_bundle_map, 
                                  axis=0)[:,np.newaxis]
        self.nonbundle_activity = np.maximum(0., (new_signals - 
                                                  combined_weights))
        self.typical_nonbundle_activity *= 1. - self.NEW_BUNDLE_UPDATE_RATE
        self.typical_nonbundle_activity += (self.nonbundle_activity * 
                                            self.NEW_BUNDLE_UPDATE_RATE)
        # As appropriate update the co-activity estimate and 
        # create new bundles
        if not self.bundles_full:
            self._create_new_bundles()
        self._update_coactivity()
        return self.bundle_activity[:self.num_bundles,:]

    def _create_new_bundles(self):
        """ If the right conditions have been reached, create a new bundle """
        # Bundle space is a scarce resource
        availability = (float(self.max_num_bundles - self.num_bundles) / 
                        float(self.max_num_bundles)) 
        new_bundle_thresholds = (self.typical_nonbundle_activity ** 2 * 
                                 availability * self.NEW_BUNDLE_FACTOR)
        cable_indices = np.where(np.random.random_sample(
                self.typical_nonbundle_activity.shape) <
                new_bundle_thresholds) 
        # Add a new bundle if appropriate
        if cable_indices[0].size > 0:
            # Randomly pick a new cable from the candidates, 
            # if there is more than one
            cable_index = cable_indices[0][int(np.random.random_sample() * 
                                                  cable_indices[0].size)]
            self.bundle_map[self.num_bundles, cable_index] = 1.
            self.num_bundles += 1
            if self.num_bundles == self.max_num_bundles:
                self.bundles_full = True
            # debug
            print self.name, 'nf', self.num_bundles, 'ni', cable_index , \
                    np.mod(cable_index,20), np.floor(cable_index/20)
            self.typical_nonbundle_activity[cable_index, 0] = 0.
        return 
          
    def _update_coactivity(self):
        """ Update an estimate of co-activity between all cables """
        instant_coactivity = np.dot(self.bundle_activity, 
                                    self.nonbundle_activity.T)
        # Determine the upper bound on the size of the incremental step 
        # toward the instant co-activity.
        delta_coactivity = instant_coactivity - self.coactivity
        bundle_coactivity_update_rate = (self.bundle_activity * 
                                         self.COACTIVITY_UPDATE_RATE)
        self.bundle_coactivity += (delta_coactivity * 
                                   bundle_coactivity_update_rate)
        cable_coactivity_update_rate = (self.nonbundle_activity.T * 
                                        self.COACTIVITY_UPDATE_RATE)
        self.cable_coactivity += (delta_coactivity * 
                                  cable_coactivity_update_rate)
        self.coactivity = np.minimum(self.bundle_coactivity,
                                     self.cable_coactivity)
        self.bundle_map[np.where(self.coactivity >= 
                                 self.JOINING_THRESHOLD)] = 1.
        return
        
    def get_estimated_cable_activities(self, bundle_activities):
        """ 
        Project the goal values to the appropriate cables

        Multiply the bundle goals across the cables that contribute 
        to them, and perform a bounded sum over all bundles to get 
        the estimated activity associated with each cable.
        """
        if bundle_activities.size > 0:
            bundle_activities = ut.pad(bundle_activities, 
                                       (self.max_num_bundles, 0))
            cable_activities = ut.bounded_sum(self.bundle_map * 
                                              bundle_activities, axis=0)
            return cable_activities
        else:
            return np.zeros((self.max_num_cables, 1))
        
    def get_projection(self, bundle_index):
        """ Project bundles down to the cables they're composed of """
        bundle = np.zeros((self.max_num_bundles, 1))
        bundle[bundle_index, 0] = 1.
        projection = np.sign(np.max(self.bundle_map * bundle, 
                                    axis=0))[np.newaxis, :]
        return projection
        
    def visualize(self, save_eps=False):
        """ Show the internal state of the map in a pictorial format """
        ut.visualize_array(self.coactivity, 
                           label=self.name + '_coactivity', 
                           save_eps=save_eps)
        ut.visualize_array(self.bundle_map, 
                           label=self.name + '_bundle_map')
        coverage = np.reshape(np.sum(self.bundle_map, axis=0), 
                              (int(np.sqrt(self.bundle_map.shape[1])), -1))
        ut.visualize_array(coverage, label=self.name + '_bundle_map_coverage')
        return
