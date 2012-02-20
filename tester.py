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

from worlds.grid_1D import Grid_1D as worldcreator
#from worlds.grid_1D_ms import Grid_1D_ms as worldcreator
#from worlds.grid_1D_noise import Grid_1D_noise as worldcreator
#from worlds.grid_2D import Grid_2D as worldcreator
#from worlds.grid_2D_dc import Grid_2D_dc as worldcreator
#from worlds.image_1D import Image_1D as worldcreator
#
#    import worlds.listen

#    import worlds.morris  # under development

#from agent.agent_stub import Agent
from agent.agent import Agent

from experiment.experiment import Experiment

def main(args):
    parser = argparse.ArgumentParser(description='tester')
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
    
    experiment = Experiment(worldcreator, Agent, pickle_prefix=parameters.pickle_prefix,
                            backup_period=parameters.save_period)
    experiment.run()

    return 0
    
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
