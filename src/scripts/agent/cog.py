import numpy as np

from ziptie import ZipTie
from daisychain import DaisyChain

class Cog(object):
    """ 
    The building block of which levels are composed

    Cogs are named for their similarity to clockwork cogwheels.
    They are simple and do the same task over and over, but by
    virtue of how they are connected to their fellows, they 
    collectively bring about interesting behavior.  

    Input channels are similar to cables in that they carry activity 
    signals that vary over time.
    Each cog contains two important parts, a daisychain and a ziptie.
    The daisychain is an object that builds cables into short sequences,
    and the ziptie is an object that takes the resulting chains
    and performs clustering on them, creating bundles.
    During upward processing, cable activities are used to train
    the daisychain and ziptie, making new bundles, maturing existing 
    bundles, and calculating the activity in bundle. 
    During downward processing, 
    the daisychain and ziptie use the bundle activity goals from 
    the next level higher to create goals for the cables. 
    """
    def __init__(self, max_cables, max_bundles, 
                 name='anonymous'):
        """ Initialize the cogs with a pre-determined maximum size """
        self.name = name
        self.max_cables = max_cables
        self.max_bundles = max_bundles
        self.daisychain = DaisyChain(max_cables, name=name)        
        if max_bundles > 0:
            self.ziptie = ZipTie(max_cables **2, max_bundles, name=name)

    def step_up(self, cable_activities, reward):
        """ cable_activities percolate upward through daisychain and ziptie """
        chain_activities = self.daisychain.update(cable_activities, reward) 
        self.reaction= self.daisychain.get_reaction()
        self.surprise = self.daisychain.get_surprise()
        bundle_activities = self.ziptie.update(chain_activities)
        return bundle_activities

    def step_down(self, bundle_activity_goals):
        """ bundle_activity_goals percolate downward """
        chain_activity_goals = self.ziptie.get_estimated_cable_activities(
                bundle_activity_goals) 
        instant_cable_activity_goals = self.daisychain.deliberate(
                chain_activity_goals)     
        self.cable_activity_goals = self.daisychain.get_goal()
        return instant_cable_activity_goals

    def get_projection(self, bundle_index):
        """ Project a bundle down through the ziptie and daisychain """
        chain_projection = self.ziptie.get_projection(bundle_index)
        cable_projection = self.daisychain.get_projection( chain_projection)
        return cable_projection
         
    def fraction_filled(self):
        """ How full is the set of cables for this cog? """
        return float(self.daisychain.num_cables) / float(self.max_cables)
        
    def visualize(self):
        """ Show the internal state of the daisychain and ziptie """
        self.daisychain.visualize()
        if self.max_bundles > 0:
            self.ziptie.visualize()
        return
