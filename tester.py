"""
Created on Jan 11, 2012

@author: brandon_rohrer

A main test harness for a general reinforcement learning agent. 
"""

import sys
import argparse
import logging

""" 
selects the World that the Agent will be placed in 
"""

Default_Save_Period = 20000

Default_World_Class = 'becca.worlds.grid_1D.Grid_1D'

from becca.agent.agent import Agent
from becca.agent.agent_stub import Agent as RandomAgent

from becca.experiment.experiment import Experiment
                    

def main(args):
    parser = argparse.ArgumentParser(description='tester')
    parser.add_argument('world_creator', nargs="?", default=Default_World_Class,
                        help="Class to create world (default: %(default)s)")
    parser.add_argument('--no-graphs', dest="graphs", default=True, action="store_false",
                        help="Don't draw tracking graphs")        
    parser.add_argument('--dummy-agent', dest="dummy_agent", default=False, action="store_true",
                        help="Use an agent that just returns random actions. Good for testing worlds")    
    parser.add_argument('-s', '--save-period', dest="save_period", type=int, default=Default_Save_Period,
                        help="How often to take snapshots (default: %(default)s)")
    parser.add_argument('-p', '--pickle-prefix', dest="pickle_prefix", default=None,
                        help="Use pickle prefix specified")
    parser.add_argument('-v', '--verbosity', dest="verbosity", default=1, type=int,
                        help="How verbose to be (default: %(default)s)")

    parameters = parser.parse_args(args)

    if parameters.verbosity <= 0:
        logging_level = logging.ERROR
    elif parameters.verbosity == 1:
        logging_level = logging.WARNING
    elif parameters.verbosity == 2:
        logging_level = logging.INFO
    elif parameters.verbosity == 3:
        logging_level = logging.DEBUG

    logging.basicConfig(level=logging_level, format="%(asctime)s %(levelname)-8s %(message)s")

    if parameters.dummy_agent:
        agent_class = RandomAgent
    else:
        agent_class = Agent

    world_creator = get_world_class(parameters.world_creator)
    if world_creator is None:
        parser.error("Couldn't find world creator %s" % parameters.world_creator)

    world = world_creator(graphs=parameters.graphs)
    agent = agent_class.FromWorld(world, graphs=parameters.graphs)
        
    experiment = Experiment(world, agent, pickle_prefix=parameters.pickle_prefix, backup_period=parameters.save_period)
    experiment.run()

    return 0


def get_world_class(world_name):
    """
    Bit of ugly magic to guess what world the user wants and load the class for it
    """

    def is_suitable_world_creator(klass):
        """
        Return whether the given klass would do as a world creator
        """
        from becca.worlds.world import World

        # do some basic checking to see if the loaded klass looks right.
        # If it's a subclass of World, then that's enough, but maybe they didn't
        # subclass from there, so just check that it is a class, and that it has a step method
        return isinstance(klass, World) or (isinstance(klass, type) and hasattr(klass, 'step'))


    def import_module(path):
        module = __import__(path)
        for sub in path.split('.')[1:]:
            module = getattr(module, sub)

        return module
    
    # option 1: properly specified dotted class name
    # eg worlds.grid_1D.Grid_1D

    klass = None
    module = None
    
    if '.' in world_name:
        path, class_name = world_name.rsplit('.', 1)
    else:
        # no dots, we'll go with the theory that it's a module in this directory, or somewhere on the path
        path = world_name
        class_name = None

    try:
        module = import_module(path)
    except ImportError:
        # nah, that didn't work. Maybe they forgot to prefix with 'worlds' and that's what they meant
        if not path.startswith('becca.worlds.'):
            path = 'becca.worlds.' + path
            try:
                module = import_module(path)
            except ImportError:
                # nah, that wasn't it either.  Give up
                pass
            else:
                # warn them
                logging.warn("Taking %s to be the proper path of the module" % path)

    if module is not None:
        if class_name is not None:
            klass = getattr(module, class_name, None)
            if not is_suitable_world_creator(klass):
                # Maybe this is a module, then we have to search for the class.
                # Try reimporting
                path += '.' + class_name
                try:
                    module = import_module(path)
                except ImportError:
                    # no. give up
                    klass = None
                else:
                    # ok, we'll search in that module for a suitable class
                    class_name = None
                    klass = None

        if class_name is None:
            logging.info("Searching in %s for suitable world creator" % module.__name__)
            # look through each class in module and see if anything in there is suitable, and defined in the module
            # itself (ie not imported from elsewhere)
            classes = [class_candidate for class_candidate in dir(module)
                       if is_suitable_world_creator(getattr(module, class_candidate, None)) and
                       getattr(module, class_candidate).__module__ == module.__name__]

            if len(classes) > 1:
                # we found many. Just tell the user we don't know which one to use
                logging.error("Found too many possible world creators in %s: %s" % (module.__name__, ', '.join(classes)))
            elif not classes:
                # we found nothing of value
                logging.error("Didn't find any clearly suitable world creator in %s. You'll have to specify it manually" % module.__name__)
            else:
                klass = getattr(module, classes[0])
    else:
        logging.error("Couldn't find module path %s" % path)

    return klass

                        


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
