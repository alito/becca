import sys

class Experiment(object):
    """
    An agent running on a world
    """

    DEFAULT_BACKUP_PERIOD = 1000
    
    def __init__(self, world, agent, pickle_prefix=None, backup_period=None):
        if backup_period is None:
            self.backup_period = self.DEFAULT_BACKUP_PERIOD
        else:
            self.backup_period = backup_period
        
        if pickle_prefix is None:
            import datetime
            now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            self.pickle_prefix = "%s_%s_%s" % (world.__class__.__name__, agent.__class__.__name__, now)
        else:
            self.pickle_prefix = pickle_prefix
            
        self.agent_pickle = self.pickle_prefix + '_agent.pickle'
        self.world_pickle = self.pickle_prefix + '_world.pickle'

        self.world = world
        pickled = self.world.load(self.world_pickle)

        self.agent = agent
        if pickled:
            agent_pickled = self.agent.load(self.agent_pickle)
            if not agent_pickled:
                print >> sys.stderr, "Well, that's awkward. The world restored, but not the agent. We should probably die"

             
    def run(self):
        import numpy
        # give an initial null action to kick things off
        actions = numpy.zeros(self.world.num_actions)
        while self.world.final_performance() < -1:
            sensors, primitives, reward = self.world.step(actions)
            actions = self.agent.step(sensors, primitives, reward)

            if (self.world.timestep % self.backup_period) == 0:
                self.world.save(self.world_pickle)
                self.agent.save(self.agent_pickle)
                
        print('final performance is {0:.3f}'.format(self.world.final_performance()))

        
