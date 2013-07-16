import numpy as np

import tools

class ZipTie(object):
    """ 
    An incremental unsupervised learning algorithm

    Input channels are clustered together into mutually co-active sets.
    A helpful metaphor is bundling cables together with zip ties.
    Cables that carry related signals are commonly co-active, 
    and are grouped together. Cables that are co-active with existing
    bundles can be added to those bundles. A single cable may be ziptied
    into several different bundles. Co-activity is estimated 
    incrementally, that is, the algorithm updates the estimate after 
    each new set of signals is received. 
    """
    # debug 
    # joining_threshold was 0.2 
    def __init__(self, max_num_cables, max_num_bundles, 
                 max_cables_per_bundle=None,
                 mean_exponent=-4, joining_threshold=0.005, 
                 speedup = 1., name='ziptie_'):
        """ Initialize each map, pre-allocating max_num_bundles """
        self.name = name
        self.max_num_cables = max_num_cables
        self.max_num_bundles = max_num_bundles
        if max_cables_per_bundle is None:
            self.max_cables_per_bundle = int(self.max_num_cables / 
                                             self.max_num_bundles)
        else:
            self.max_cables_per_bundle = max_cables_per_bundle
        self.num_bundles = 0
        # User-defined constants
        #
        # real, 0 < x < 1, small
        self.COACTIVITY_UPDATE_RATE = 10 ** -4 * speedup
        # The rate at which the nonbundle_activities thresholds are updated
        # real, 0 < x < 1, small
        self.NONBUNDLE_ACTIVITY_UPDATE_RATE = 4 * 10 ** -4 * speedup
        # Constant factor driving the rate at which new bundles are created
        # real, 0 < x < 1, small
        self.NEW_BUNDLE_FACTOR = 10 ** -5
        # Coactivity value which, if it's every exceeded, causes a 
        # cable to be added to a bundle
        # real, 0 < x < 1, small
        self.JOINING_THRESHOLD = joining_threshold
        # Exponent for calculating the generalized mean of signals in 
        # order to find bundle activities
        # real, x != 0
        self.MEAN_EXPONENT = mean_exponent
        
        self.bundles_full = False        
        self.bundle_activities = np.zeros((self.max_num_bundles, 1))
        map_size = (self.max_num_bundles, self.max_num_cables)
        self.bundle_map = np.zeros(map_size)
        self.bundle_coactivity = np.zeros(map_size)
        self.cable_coactivity = np.zeros(map_size)
        self.coactivity = np.zeros(map_size)
        self.typical_nonbundle_activity = np.zeros((self.max_num_cables, 1))

    def update(self, cable_activities):
        """ Update co-activity estimates and calculate bundle activity """
        # Find bundle activities by taking the generalized mean of
        # the signals with a negative exponent.
        # The negative exponent weights the lower signals more heavily.
        # At the extreme value of negative infinity, the 
        # generalized mean becomes the minimum operator.
        self.cable_activities = cable_activities
        shifted_cable_activities = cable_activities + 1.
        activities_to_power = shifted_cable_activities ** self.MEAN_EXPONENT
        mean_activities_to_power = tools.weighted_average(activities_to_power,
                                                          self.bundle_map.T)
        shifted_bundle_activities = (mean_activities_to_power + 
                                     tools.EPSILON) ** (1./self.MEAN_EXPONENT)
        self.bundle_activities[:self.num_bundles,:] = (
                shifted_bundle_activities - 1.)[:self.num_bundles]
        # debug
        # Winner take all within the ziptie
        # TODO: Move to a cable-energy approach, rather than the crude
        # winner take all. WTA is probably appropriate for
        # small cogs, but not necessarily so for larger cogs in which
        # multiple disjoint features may be active simultaneously.
        #self.bundle_activities[np.where(self.bundle_activities != 
        #                       np.max(self.bundle_activities))] = 0.
        
        # Calculate how much energy each signal has left to contribute 
        # to the co-activity estimate
        final_activated_bundle_map = self.bundle_activities * self.bundle_map
        combined_weights = np.sum(final_activated_bundle_map, 
                                  axis=0)[:,np.newaxis]
        self.nonbundle_activities = np.maximum(0., (cable_activities - 
                                                    combined_weights))
        self.typical_nonbundle_activity *= (
                1. - self.NONBUNDLE_ACTIVITY_UPDATE_RATE)
        self.typical_nonbundle_activity += (
                self.nonbundle_activities *self.NONBUNDLE_ACTIVITY_UPDATE_RATE)
        # As appropriate update the co-activity estimate and 
        # create new bundles
        if not self.bundles_full:
            self._create_new_bundles()
        self._update_coactivity()
        return self.bundle_activities[:self.num_bundles,:]

    def _create_new_bundles(self):
        """ If the right conditions have been reached, create a new bundle """
        # Bundle space is a scarce resource
        availability = (float(self.max_num_bundles - self.num_bundles) / 
                        float(self.max_num_bundles)) 
        # debug
        #print self.name, 'tna', np.max(self.typical_nonbundle_activity)
        new_bundle_thresholds = (self.typical_nonbundle_activity ** 
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
            self.typical_nonbundle_activity[cable_index, 0] = 0.
        return 
          
    def _update_coactivity(self):
        """ Update an estimate of co-activity between all cables """
        instant_coactivity = np.dot(self.bundle_activities, 
                                    self.nonbundle_activities.T)
        # Determine the upper bound on the size of the incremental step 
        # toward the instant co-activity.
        delta_coactivity = instant_coactivity - self.coactivity
        bundle_coactivity_update_rate = (self.bundle_activities * 
                                         self.COACTIVITY_UPDATE_RATE)
        self.bundle_coactivity += (delta_coactivity * 
                                   bundle_coactivity_update_rate)
        cable_coactivity_update_rate = (self.nonbundle_activities.T * 
                                        self.COACTIVITY_UPDATE_RATE)
        self.cable_coactivity += (delta_coactivity * 
                                  cable_coactivity_update_rate)
        self.coactivity = np.minimum(self.bundle_coactivity,
                                     self.cable_coactivity)
        
        # for any bundles that are already full, don't change their coactivity
        # TODO: make this more elegant than enforcing a hard maximum count
        full_bundles = np.zeros((self.max_num_bundles, 1))
        cables_per_bundle = np.sum(self.bundle_map, axis=1)[:,np.newaxis]
        full_bundles[np.where(cables_per_bundle >= 
                              self.max_cables_per_bundle)] = 1.
        self.coactivity *= 1. - full_bundles
        
	new_candidates = np.where(self.coactivity >= self.JOINING_THRESHOLD)
	num_candidates =  new_candidates[0].size 
	if num_candidates > 0:
	    candidate_index = np.random.randint(num_candidates) 
	    self.bundle_map[new_candidates[0][candidate_index],
			    new_candidates[1][candidate_index]]  = 1.
	    #self.bundle_map[np.where(self.coactivity >= 
        #                         self.JOINING_THRESHOLD)] = 1.
        return
        
    def get_cable_deliberation_vote(self, bundle_activity_goals):
        """ 
        Project the bundle goal values to the appropriate cables

        Multiply the bundle goals across the cables that contribute 
        to them, and perform a bounded sum over all bundles to get 
        the estimated activity associated with each cable.
        """
        if bundle_activity_goals.size > 0:
            bundle_activity_goals = tools.pad(bundle_activity_goals, 
                                       (self.max_num_bundles, 0))
            cable_activity_goals = tools.bounded_sum(self.bundle_map * 
                                              bundle_activity_goals, axis=0)
        else:
            cable_activity_goals = np.zeros((self.max_num_cables, 1))
        return cable_activity_goals
        
    def get_projection(self, bundle_index):
        """ Project bundles down to the cables they're composed of """
        bundle = np.zeros((self.max_num_bundles, 1))
        bundle[bundle_index, 0] = 1.
        projection = np.sign(np.max(self.bundle_map * bundle, 
                                    axis=0))[np.newaxis, :]
        return projection
        
    def visualize(self, save_eps=False):
        """ Show the internal state of the map in a pictorial format """
        # debug
        #tools.visualize_array(self.coactivity, 
        #                   label=self.name + '_coactivity', 
        #                   save_eps=save_eps)
        #tools.visualize_array(self.bundle_map, 
        #                   label=self.name + '_bundle_map')
        print self.name, '0', np.nonzero(self.bundle_map)[0]
        print self.name, '1', np.nonzero(self.bundle_map)[1]
        return
