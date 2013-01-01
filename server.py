"""
A TCP server (text I/O) interface for BECCA that can be used to interface with other software
"""

import sys
import struct
import thread
import numpy as np

from agent.agent import Agent
        

from worlds.base_world import World as BaseWorld

#log = open("/tmp/beccalog","w")

class ShellWorld(BaseWorld):

    def __init__(self, num_sensors, num_primitives, num_actions):
                
        super(BaseWorld, self).__init__()
        

        self.num_sensors = num_sensors
        self.num_primitives = num_primitives
        self.num_actions = num_actions
        self.timestep = 0

    
    def step(self): 
        """ Advance the World by one timestep """
        
        self.timestep += 1 
       
       
        #self.display()
        
        #return sensors, primitives, reward
    
        

    def set_agent_parameters(self, agent):
        pass
        
        
    def display(self):
        """ Provide an intuitive display of the current state of the World 
        to the user.
        """
        #...



#def step():

def init(numSensors, numPrimitives, numActions, numFeatures):
    global world, agent, actions

    world = ShellWorld(numSensors, numPrimitives, numActions)
    
    """ A unique identifying string for the agent, allowing specific
    saved agents to be recalled. 
    """
    agent_name = "test";

    agent = Agent(world.num_sensors, world.num_primitives, 
                  world.num_actions, numFeatures, agent_name)
    
    agent.display_state = False
    agent.REPORTING_PERIOD = 10**4

    """ Control how rapidly previous inputs are forgotten """
    agent.perceiver.INPUT_DECAY_RATE = 0.5 # real, 0 < x < 1

    """ Control how rapidly the coactivity update platicity changes """
    agent.perceiver.PLASTICITY_UPDATE_RATE = 4 * 10 ** (-1)

    agent.perceiver.NEW_GROUP_THRESHOLD = 0.25
    agent.perceiver.MAX_PLASTICITY = 0.1

    agent.actor.WORKING_MEMORY_DECAY_RATE = 0.5      # real, 0 < x <= 1

    """ If uncommented, try to restore the agent from saved data.
    If commented out, start fresh each time.
    """
    #agent = agent.restore()
    actions = np.zeros(world.num_actions)
    
    """ If configured to do so, the world sets some Becca parameters to 
    modify its behavior. This is a development hack, and should eventually be 
    removed as Becca matures and settles on a good, general purpose
    set of parameters.
    """
    world.set_agent_parameters(agent)
         
    """ Report the performance of the agent on the world. """
    #agent.report_performance()
    #agent.show_reward_history()
    
    
def plots():
    global agent
    print 'Updating plots'
    agent.record_reward_history()
    agent.show_reward_history()
    agent.perceiver.visualize()
    agent.actor.visualize()


def act(sensors, reward):
    global world, agent, actions
        world.step()
        actions = agent.step(np.array(sensors), actions, reward)
    #agent.report_performance()
    #agent.show_reward_history()
    return actions.tolist()

import SocketServer
SocketServer.TCPServer.allow_reuse_address = True 


class MyTCPHandler(SocketServer.StreamRequestHandler):
    def setup(self):
        self.connection = self.request
        if self.timeout is not None:
            self.connection.settimeout(self.timeout)
        if self.disable_nagle_algorithm:
            self.connection.setsockopt(socket.IPPROTO_TCP,
	                           socket.TCP_NODELAY, True)
        self.rfile = self.connection.makefile('rb', self.rbufsize)
        self.wfile = self.connection.makefile('wb', self.wbufsize)

    def handle(self):
        #inputBufferSize = 1024*128 #in bytes

        # self.request is the TCP socket connected to the client
        #self.data = self.request.recv(inputBufferSize).strip()

        serving = True
        while (serving):
            self.data = self.rfile.readline().strip()
            if (self.data == ""):
                serving = False
                break
    
        #TODO Security - validate command to avoid allowing invocation of arbitrary commands
        #either attached as an extra command with ':', or invoked via a parameter to these functions
        c = self.data
        if (c.startswith('act(') or c.startswith('init(') or c.startswith('plots(')):
            v = eval(self.data)
            self.wfile.write(str(v))
            self.wfile.write('\n')
            self.wfile.flush()


def serve():

    HOST, PORT = "localhost", 9999

    # Create the server, binding to localhost on port 9999
    server = SocketServer.TCPServer((HOST, PORT), MyTCPHandler)

    # Activate the server; this will keep running until you
    # interrupt the program with Ctrl-C

    server.serve_forever()

if __name__ == "__main__":
    serve()

