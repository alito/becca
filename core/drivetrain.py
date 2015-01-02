""" the Drivetrain class """
import numpy as np
import gearbox

class Drivetrain(object):
    """
    The collection of gearboxes that form the backbone of the agent

    A drivetrain contains a hierarchical series of gearboxes. 
    The drivetrain performs two functions, 
    1) a step_up 
    where sensor activities are passed up the drivetrain and processed 
    into increasing abstract feature activities
    2) a step_down 
    where goals are passed down the drivetrain and processed 
    into increasingly concrete actions.
    """
    def __init__(self, min_cables):
        """ 
        Initialize the drivetrain.

        min_cables is the minimum number of cables that a gearbox in the
        drivetrain should be able to accomodate.
        """
        self.num_gearboxes =  1
        self.min_cables = min_cables
        first_gearbox_name = ''.join(('gearbox_', str(self.num_gearboxes - 1)))
        self.gearboxes = [gearbox.Gearbox(self.min_cables, 
                                          name=first_gearbox_name)]
        self.cables_per_gearbox = self.gearboxes[0].max_cables
        self.bundles_per_gearbox = self.gearboxes[0].max_bundles
        self.gearbox_added = False
        self.surprise_history = []
        self.recent_surprise_history = [0.] * 100

    def step_up(self, action, sensors):
        """ Find feature_activities that result from new cable_activities """
        self.num_actions = action.size
        cable_activities = np.vstack((action, sensors))
        for gearbox in self.gearboxes:
            cable_activities = gearbox.step_up(cable_activities) 
        # Create a new gearbox if the top gearbox has had 
        # enough bundles assigned
        if gearbox.bundles_created() > 1:
            self.add_gearbox()
            cable_activities = self.gearboxes[-1].step_up(cable_activities) 
        # Build full feature activities array
        num_features = self.cables_per_gearbox * len(self.gearboxes)
        feature_activities = np.zeros((num_features , 1))
        for (gearbox_index, gearbox) in enumerate(self.gearboxes):
            start_index = self.cables_per_gearbox * gearbox_index
            end_index = self.cables_per_gearbox * (gearbox_index + 1)
            feature_activities[start_index: end_index] = \
                    gearbox.cable_activities.copy()
        return feature_activities

    def assign_goal(self, goal_index): 
        """
        Assign goal to the appropriate gearbox
        
        When a goal cable is selected by the hub, it doesn't know which
        gearbox it belongs to. This method sorts that out. 
        """
        gearbox_index = int(np.floor(goal_index / self.cables_per_gearbox))
        cable_index = goal_index - gearbox_index * self.cables_per_gearbox
        # Activate the goal
        self.gearboxes[gearbox_index].hub_cable_goals[cable_index] = 1.

    def step_down(self):
        """ Find the primitive actions driven by a set of goals """
        # Propogate the deliberation_goal_votes down through the gearboxes
        agent_surprise = 0.0
        cable_goals = np.zeros((self.bundles_per_gearbox, 1))
       
        for gearbox in reversed(self.gearboxes):
            cable_goals = gearbox.step_down(cable_goals)
            if np.nonzero(gearbox.surprise)[0].size > 0:
                agent_surprise = np.sum(gearbox.surprise)
        # Tabulate and record the surprise registered in each gearbox
        self.recent_surprise_history.pop(0)
        self.recent_surprise_history.append(agent_surprise)
        self.typical_surprise = np.median(np.array(
                self.recent_surprise_history))
        mod_surprise = agent_surprise - self.typical_surprise
        self.surprise_history.append(mod_surprise)
        # Report the action that resulted for the current time step.
        # Strip the actions off the cable_goals to make 
        # the current set of actions.
        action = cable_goals[:self.num_actions,:] 
        return action 

    def add_gearbox(self):
        """ When the last gearbox creates its first bundle, add a gearbox """
        next_gearbox_name = ''.join(('gearbox_', str(self.num_gearboxes)))
        self.gearboxes.append(gearbox.Gearbox(self.cables_per_gearbox,
                                 name=next_gearbox_name, 
                                 level=self.num_gearboxes))
        print "Added gearbox", self.num_gearboxes
        self.num_gearboxes +=  1
        self.gearbox_added = True

    def get_index_projections(self, to_screen=False):
        """
        Get representations of all the bundles in each gearbox 
        
        Every feature is projected down through its own gearbox and
        the gearboxes below it until its cable_contributions on sensor inputs 
        and actions is obtained. This is a way to represent the
        receptive field of each feature.

        Returns a list containing the cable_contributions for each feature 
        in each gearbox.
        """
        all_projections = []
        all_bundle_activities = []
        for gearbox_index in range(len(self.gearboxes)):
            gearbox_projections = []
            gearbox_bundle_activities = []
            num_bundles = self.gearboxes[gearbox_index].max_bundles
            for bundle_index in range(num_bundles):    
                bundles = np.zeros((num_bundles, 1))
                bundles[bundle_index, 0] = 1.
                cable_contributions = self._get_index_projection(
                        gearbox_index,bundles)
                if np.nonzero(cable_contributions)[0].size > 0:
                    gearbox_projections.append(cable_contributions)
                    gearbox_bundle_activities.append(
                            self.gearboxes[gearbox_index].
                            bundle_activities[bundle_index])
                    # Display the cable_contributions in text form if desired
                    if to_screen:
                        print 'cable_contributions', \
                            self.gearboxes[gearbox_index].name, \
                            'feature', bundle_index
                        for i in range(cable_contributions.shape[1]):
                            print np.nonzero(cable_contributions)[0][
                                    np.where(np.nonzero(
                                    cable_contributions)[1] == i)]
            if len(gearbox_projections) > 0:
                all_projections.append(gearbox_projections)
                all_bundle_activities.append(gearbox_bundle_activities)
        return (all_projections, all_bundle_activities)

    def _get_index_projection(self, gearbox_index, bundles):
        """
        Get the cable_contributions for bundles
        
        Recursively project bundles down through gearboxes
        until the bottom gearbox is reached. Feature values is a 
        two-dimensional array and can contain
        several columns. Each column represents a state, and their
        order represents a temporal progression. During cable_contributions
        to the next lowest gearbox, the number of states
        increases by one. 
        
        Return the cable_contributions in terms of basic sensor 
        inputs and actions. 
        """
        if gearbox_index == -1:
            return bundles
        time_steps = bundles.shape[1] 
        cable_contributions = np.zeros((self.gearboxes[gearbox_index].max_cables, 
                                        time_steps * 2))
        for bundle_index in range(bundles.shape[0]):
            for time_index in range(time_steps):
                if bundles[bundle_index, time_index] > 0:
                    new_contribution = self.gearboxes[
                            gearbox_index].get_index_projection(bundle_index)
                    cable_contributions[:, 2*time_index: 2*time_index + 2] = ( 
                            np.maximum(cable_contributions[:, 
                            2*time_index: 2*time_index + 2], new_contribution))
        cable_contributions = self._get_index_projection(gearbox_index - 1, 
                                                   cable_contributions)
        return cable_contributions

    def visualize(self):
        print 'drivetrain:'
        for gearbox in self.gearboxes:
            gearbox.visualize()
