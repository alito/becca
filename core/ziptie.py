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
    def __init__(self, max_num_cables, max_num_bundles, 
                 max_cables_per_bundle=None,
                 mean_exponent=-4, joining_threshold=0.05, 
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
        #self.COACTIVITY_UPDATE_RATE = 10 ** -4 * speedup
        self.AGGLOMERATION_ENERGY_RATE = 10 ** -2 * speedup
        # The rate at which affinity is updated
        # Affinity is the potentiation of a cable to being included in a
        # bundle. It increases over time when the cable is inactive and
        # decreases when the cable is active. 
        #self.AFFINITY_UPDATE_RATE = 10 ** -2
        # The rate at which the nonbundle_activities thresholds are updated
        # real, 0 < x < 1, small
        #self.NONBUNDLE_ACTIVITY_UPDATE_RATE = 4 * 10 ** -4 * speedup
        # Constant factor driving the rate at which new bundles are created
        # real, 0 < x < 1, small
        #self.NEW_BUNDLE_FACTOR = 10 ** -5
        self.NUCLEATION_ENERGY_RATE = 10 ** -4 * speedup
        self.ENERGY_DECAY_RATE = 10 ** -2
        # Coactivity value which, if it's every exceeded, causes a 
        # cable to be added to a bundle
        # real, 0 < x < 1, small
        self.JOINING_THRESHOLD = joining_threshold
        self.NUCLEATION_THRESHOLD = joining_threshold
        # Exponent for calculating the generalized mean of signals in 
        # order to find bundle activities
        # real, x != 0
        self.MEAN_EXPONENT = mean_exponent
        # Exponent controlling the strength of inhibition between bundles
        self.ACTIVATION_WEIGHTING_EXPONENT = 6.

        self.bundles_full = False        
        self.bundle_activities = np.zeros((self.max_num_bundles, 1))
        #self.affinity = np.ones((self.max_num_cables, 1))
        map_size = (self.max_num_bundles, self.max_num_cables)
        self.bundle_map = np.zeros(map_size)
        #self.bundle_coactivities = np.zeros(map_size)
        #self.cable_coactivities = np.zeros(map_size)
        #self.coactivities = np.zeros(map_size)
        self.agglomeration_energy = np.zeros(map_size)
        #self.typical_nonbundle_activities = np.zeros((self.max_num_cables, 1))
        self.nucleation_energy = np.zeros((self.max_num_cables, 1))

    def step_up(self, cable_activities):
        """ Update co-activity estimates and calculate bundle activity """
        # Find bundle activities by taking the generalized mean of
        # the signals with a negative exponent.
        # The negative exponent weights the lower signals more heavily.
        # At the extreme value of negative infinity, the 
        # generalized mean becomes the minimum operator.
        # Shifting by one helps handle zero-valued cable activities.
        self.cable_activities = cable_activities
        # debug
        #print np.mean(self.cable_activities)
        # weighted mean 
        #initial_bundle_activities_gm = tools.generalized_mean(
        #        self.cable_activities, self.bundle_map.T, 1)
        initial_bundle_activities = tools.generalized_mean(
                self.cable_activities, self.bundle_map.T, self.MEAN_EXPONENT)
        # debug      
        # Make a first pass at the bundle activation levels by 
        # multiplying across the bundle map.

        #print self.name
        #print 'ca', self.cable_activities.shape
        #print 'ca', self.cable_activities.ravel()
        #print 'nb', self.num_bundles
        #print 'bm', self.bundle_map.shape
        #print 'bm-sized', self.bundle_map [:self.num_bundles, 
        #                                   :self.cable_activities.size] 
       
        #initial_bundle_activities = np.dot(self.bundle_map,
        #                                   self.cable_activities)
        #print 'iba', initial_bundle_activities.shape
        #print self.name
        #print 'ibag', initial_bundle_activities_gm.ravel()
        #print 'iba', initial_bundle_activities.ravel()
        #print 'diba', (initial_bundle_activities_gm - initial_bundle_activities).ravel()
        # Find the activity levels of the bundles contributed to 
        # by each cable.
        
        bundle_contribution_map = np.zeros(self.bundle_map.shape)
        #bundle_contribution_map = np.zeros((self.num_bundles, 
        #                                    self.cable_activities.size))
        bundle_contribution_map[np.nonzero(self.bundle_map)] = 1.
        #print 'bcm', bundle_contribution_map.shape
        activated_bundle_map = (initial_bundle_activities * 
                                bundle_contribution_map)
        #print 'abm', activated_bundle_map.shape                                                
        # Find the largest bundle activity that each input contributes to
        max_activation = np.max(activated_bundle_map, axis=0) + tools.EPSILON
        #print 'ma', max_activation
        # Divide the energy that each input contributes to each bundle
        input_inhibition_map = np.power(activated_bundle_map / max_activation, 
                                        self.ACTIVATION_WEIGHTING_EXPONENT)
        #print 'iim', input_inhibition_map.shape
        # Find the effective strength of each cable to each bundle 
        # after inhibition.
        inhibited_cable_activities = (input_inhibition_map * 
                                      self.cable_activities.T)
        #print 'ii', inhibited_cable_activities.shape
        #print 'bm', self.bundle_map.shape
        #print 'di', np.nonzero(inhibited_cable_activities)
        final_bundle_activities = tools.generalized_mean(
                inhibited_cable_activities.T, self.bundle_map.T, 
                self.MEAN_EXPONENT)
        #final_bundle_activities = np.sum(self.bundle_map * 
        #                                 inhibited_cable_activities, axis=1)
        #final_bundle_activities = np.sum(
        #        self.bundle_map[:self.num_bundles,:self.cable_activities.size] *
        #        inhibited_cable_activities, axis=1)
        #print 'fba', final_bundle_activities.shape
        #print 'fba', final_bundle_activities.ravel()
        #self.bundle_activities = final_bundle_activities[:,np.newaxis]
        self.bundle_activities = final_bundle_activities
        #self.bundle_activities[:self.num_bundles,0] = (
        #        final_bundle_activities[:self.num_bundles])
        #print 'ba', self.bundle_activities.shape
        # Calculate how much energy each input has left to contribute 
        # to the co-activity estimate. 
        final_activated_bundle_map = (final_bundle_activities * 
                                      bundle_contribution_map)

        #final_activated_bundle_map = (final_bundle_activities[:,np.newaxis] * 
        #                              bundle_contribution_map)
        #print 'fabm', final_activated_bundle_map.shape
        #combined_weights = (np.sum(final_activated_bundle_map, axis=0) + 
        #                    tools.EPSILON)
        # combination method 1: exponential weighting
        #coactivities_inputs = self.cable_activities * 2 ** (
        #    -combined_weights[:, np.newaxis] * self.DISSIPATION_FACTOR)
        
        # Calculate how much energy each signal has left to contribute 
        # to the co-activities estimate
        #final_activated_bundle_map = self.bundle_activities * self.bundle_map
        # combination method 2: straight sum
        combined_weights = np.sum(final_activated_bundle_map, 
                                  axis=0)[:,np.newaxis]
        #print 'cw', combined_weights.shape
        #print 'cw', combined_weights.ravel()
        self.nonbundle_activities = np.maximum(0., (cable_activities - 
                                                    combined_weights))
        #self.typical_nonbundle_activities *= (
        #        1. - self.NONBUNDLE_ACTIVITY_UPDATE_RATE)
        #self.typical_nonbundle_activities += (
        #        self.nonbundle_activities *self.NONBUNDLE_ACTIVITY_UPDATE_RATE)
        #print 'tnba', self.typical_nonbundle_activities.ravel()
        # As appropriate update the co-activity estimate and 
        # create new bundles
        if not self.bundles_full:
            self._create_new_bundles()
        self._grow_bundles()
        return self.bundle_activities[:self.num_bundles,:]

    def _create_new_bundles(self):
        """ If the right conditions have been reached, create a new bundle """
        # Bundle space is a scarce resource
        #debug
        #availability = (float(self.max_num_bundles - self.num_bundles) / 
        #                float(self.max_num_bundles)) 
        availability = 1.
        # debug
        #print self.name, 'tna', np.max(self.typical_nonbundle_activities)
        #new_bundle_thresholds = (self.typical_nonbundle_activities ** 
        #                         availability * self.NEW_BUNDLE_FACTOR)
        #cable_indices = np.where(np.random.random_sample(
        #        self.typical_nonbundle_activities.shape) <
        #        new_bundle_thresholds) 
        # Decay the energy        
        self.nucleation_energy -= (self.cable_activities *
                                   self.nucleation_energy * 
                                   self.NUCLEATION_ENERGY_RATE * 
                                   self.ENERGY_DECAY_RATE)
        self.nucleation_energy += (self.nonbundle_activities * 
                                   (1. - self.nucleation_energy) *
                                   self.NUCLEATION_ENERGY_RATE)
        #print 'nba', self.nonbundle_activities.ravel()
        #print 'ne', self.nucleation_energy.ravel()
        cable_indices = np.where(self.nucleation_energy * availability > 
                                 self.NUCLEATION_THRESHOLD)
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
            #self.typical_nonbundle_activities[cable_index, 0] = 0.
            print self.name, 'ci', cable_index, 'added as a bundle nucleus'
            self.nucleation_energy[cable_index, 0] = 0.
            self.agglomeration_energy[:, cable_index] = 0.
        return 
          
    def _grow_bundles(self):
        """ Update an estimate of co-activity between all cables """
        #print self.name
        #print 'ba', self.bundle_activities.shape
        #print 'nba', self.nonbundle_activities.shape
        coactivities = np.dot(self.bundle_activities, 
                              self.nonbundle_activities.T)
        #print 'ca', coactivities.shape
        #print coactivities
        # Each cable's nonbundle activity is distributed to agglomeration energy 
        # with each bundle proportionally to their coactivities.
        proportions_by_bundle = (self.bundle_activities / 
                                 np.sum(self.bundle_activities + tools.EPSILON))
        proportions_by_cable = np.dot(proportions_by_bundle, 
                                      self.nonbundle_activities.T)
        #print 'ppb', proportions_by_bundle.ravel()
        #print 'ppc', proportions_by_cable.ravel()
        # Decay the energy        
        self.agglomeration_energy -= (proportions_by_cable * 
                                      self.cable_activities.T *
                                      self.agglomeration_energy * 
                                      self.AGGLOMERATION_ENERGY_RATE * 
                                      self.ENERGY_DECAY_RATE)
        self.agglomeration_energy += (proportions_by_cable * 
                                      coactivities * 
                                      (1. - self.agglomeration_energy) *
                                      self.AGGLOMERATION_ENERGY_RATE)
        #print self.agglomeration_energy
        # Determine the upper bound on the size of the incremental step 
        # toward the instant co-activity.
        #instant_coactivities = np.dot(self.bundle_activities, 
        #                              self.nonbundle_activities.T)
        #delta_coactivities = instant_coactivities - self.coactivities
        #bundle_coactivities_update_rate = (self.bundle_activities * 
        #                                 self.COACTIVITY_UPDATE_RATE)
        #self.bundle_coactivities += (delta_coactivities * 
        #                           bundle_coactivities_update_rate)
        #debug
        #print 'nba', self.nonbundle_activities.shape
        #print 'aff', self.affinity.shape
        #print 'ca', self.cable_activities.shape
        #print 'ca_full', self.cable_activities.ravel()

        #cable_coactivities_update_rate = (self.nonbundle_activities.T *
        #                                self.COACTIVITY_UPDATE_RATE)
        #cable_coactivities_update_rate = (self.nonbundle_activities.T *
        #                                self.affinity.T * 
        #                                self.COACTIVITY_UPDATE_RATE)
        #self.cable_coactivities += (delta_coactivities * 
        #                          cable_coactivities_update_rate)
        #self.coactivities = np.minimum(self.bundle_coactivities,
        #                               self.cable_coactivities)
        # Update affinity
        #self.affinity *= 1. - self.cable_activities
        #self.affinity += 1. - self.affinity * self.AFFINITY_UPDATE_RATE
        #print "affinity", self.affinity.ravel()
        # debug
        # Turn off affinity
        #self.affinity = np.ones(self.affinity.shape)
        
        
        # for any bundles that are already full, don't change their coactivity
        # TODO: make this more elegant than enforcing a hard maximum count
        full_bundles = np.zeros((self.max_num_bundles, 1))
        cables_per_bundle = np.sum(self.bundle_map, axis=1)[:,np.newaxis]
        full_bundles[np.where(cables_per_bundle >= 
                              self.max_cables_per_bundle)] = 1.
        #self.coactivities *= 1. - full_bundles
        self.agglomeration_energy *= 1 - full_bundles

        #new_candidates = np.where(self.coactivities >= self.JOINING_THRESHOLD)
        new_candidates = np.where(self.agglomeration_energy >= 
                                  self.JOINING_THRESHOLD)
        num_candidates =  new_candidates[0].size 
        if num_candidates > 0:
            candidate_index = np.random.randint(num_candidates) 
            candidate_cable = new_candidates[1][candidate_index]
            candidate_bundle = new_candidates[0][candidate_index]
            self.bundle_map[candidate_bundle, candidate_cable] = 1.
            #self.bundle_map[np.where(self.coactivities >= 
            #                         self.JOINING_THRESHOLD)] = 1.
            self.nucleation_energy[candidate_cable, 0] = 0.
            self.agglomeration_energy[:, candidate_cable] = 0.
            print self.name, 'cable', candidate_cable, 'added to bundle', \
                    candidate_bundle
        return
        
    def step_down(self, bundle_goals):
        """ 
        Project the bundle goal values to the appropriate cables

        Multiply the bundle goals across the cables that contribute 
        to them, and perform a bounded sum over all bundles to get 
        the estimated activity associated with each cable.
        """
        if bundle_goals.size > 0:
            bundle_goals = tools.pad(bundle_goals, (self.max_num_bundles, 0))
            cable_activity_goals = tools.bounded_sum(self.bundle_map * 
                                                     bundle_goals, axis=0)
        else:
            cable_activity_goals = np.zeros((self.max_num_cables, 1))
        return cable_activity_goals
        
    def get_index_projection(self, bundle_index):
        """ Project bundle indices down to their cable indices """
        bundle = np.zeros((self.max_num_bundles, 1))
        bundle[bundle_index, 0] = 1.
        projection = np.sign(np.max(self.bundle_map * bundle, 
                                    axis=0))[np.newaxis, :]
        return projection
        
    def cable_fraction_in_bundle(self, bundle_index):
        cable_count = np.nonzero(self.bundle_map[bundle_index,:])[0].size
        cable_fraction = float(cable_count) / float(self.max_cables_per_bundle)
        return cable_fraction

    def visualize(self, save_eps=False):
        """ Show the internal state of the map in a pictorial format """
        # debug
        #tools.visualize_array(self.coactivities, 
        #                   label=self.name + '_coactivities', 
        #                   save_eps=save_eps)
        #tools.visualize_array(self.bundle_map, 
        #                   label=self.name + '_bundle_map')
        print self.name, '0', np.nonzero(self.bundle_map)[0]
        print self.name, '1', np.nonzero(self.bundle_map)[1]
        print self.max_num_bundles, 'bundles maximum'
        return
