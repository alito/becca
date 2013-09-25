import numpy as np

from daisychain import DaisyChain
import tools
from ziptie import ZipTie

class Cog(object):
    """ 
    The basic units of which blocks are composed

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
    def __init__(self, max_cables, max_bundles, max_chains_per_bundle=None,
                 name='anonymous', level=0):
        """ Initialize the cogs with a pre-determined maximum size """
        self.name = name
        self.max_cables = max_cables
        self.max_bundles = max_bundles
        if max_chains_per_bundle is None:
            max_chains_per_bundle = int(max_cables ** 2 / max_bundles)
        self.daisychain = DaisyChain(max_cables, name=name)        
        if max_bundles > 0:
            self.ziptie = ZipTie(max_cables **2, max_bundles, 
                                 max_cables_per_bundle=max_chains_per_bundle, 
                                 name=name)

    def step_up(self, cable_activities, reward, enough_cables):
        # TODO: fix this so that cogs can gracefully handle more cables 
        # or else never be assigned them in the first place
        if cable_activities.size > self.max_cables:
            cable_activities = cable_activities[:self.max_cables, :]
            print '-----  Number of max cables exceeded in', self.name, \
                    '  -----'
        """ cable_activities percolate upward through daisychain and ziptie """
        chain_activities = self.daisychain.update(cable_activities, reward) 
        self.reaction= self.daisychain.get_cable_activity_reactions()
        self.surprise = self.daisychain.get_surprise()
        if enough_cables is True:
            bundle_activities = self.ziptie.update(chain_activities)
        else:
            bundle_activities = np.zeros((0,1))
        bundle_activities = tools.pad(bundle_activities, (self.max_bundles, 0))
        return bundle_activities

    def step_down(self, bundle_activity_goals):
        """ bundle_activity_goals percolate downward """
        chain_activity_goals = self.ziptie.get_cable_deliberation_vote(
                bundle_activity_goals) 
        instant_cable_activity_goals = self.daisychain.deliberate(
                chain_activity_goals)     
        self.cable_activity_goals =self.daisychain.get_cable_deliberation_vote()
        return instant_cable_activity_goals

    def get_projection(self, bundle_index):
        """ Project a bundle down through the ziptie and daisychain """
        chain_projection = self.ziptie.get_projection(bundle_index)
        cable_projection = self.daisychain.get_projection(chain_projection)
        return cable_projection
         
    def fraction_filled(self):
        """ How full is the set of cables for this cog? """
        return float(self.daisychain.num_cables) / float(self.max_cables)

    def num_bundles(self):
        """ How many bundles have been created in this cog? """
        return self.ziptie.num_bundles
            
    def visualize(self):
        """ Show the internal state of the daisychain and ziptie """
        self.daisychain.visualize()
        if self.max_bundles > 0:
            self.ziptie.visualize()
        return
