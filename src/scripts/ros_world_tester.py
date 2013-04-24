#!/usr/bin/env python
import rospy
from rospy.numpy_msg import numpy_msg
from becca_world_test.msg import world_results
from becca_world_test.msg import agent_commands


import numpy as np
        
# Select the World that the Agent will be placed in. 
# Only one of these import lines should be uncommented. 
#from worlds.base_world import World
from rosworlds.grid_1D import World
#from worlds.grid_1D_ms import World
#from worlds.grid_1D_noise import World
#from worlds.grid_2D import World
#from worlds.grid_2D_dc import World
#from worlds.image_1D import World
#from worlds.image_2D import World

""" If you want to run a world of your own, add the appropriate line here """
#from worlds.hello import World
#from becca_world_listen.listen import World


class test_ros_world:
    def __init__(self, world):
        self.world = world
        self.world_pub = rospy.Publisher('world_results', numpy_msg(world_results))
        rospy.init_node('world1')
        rospy.Subscriber("agent_commands", numpy_msg(agent_commands), self.callback)
        self.actions = np.zeros((world.num_actions,1))
    
    def step(self, actions):
        return self.world.step(actions)
    
    def spin(self):
        print "spinning"
        rospy.spin()

    def callback(self, data):
        print "got ", data
        actions = [[x] for x in data.action]
        sensors, reward = self.step(np.array(actions))
        self.world_pub.publish(sensors, reward)
  

if __name__ == '__main__':
    global tr_world
    tr_world=test_ros_world(World(lifespan=10**6))
    sensors, reward = tr_world.step(tr_world.actions)
    print sensors, reward
    print sensors, type(sensors), type(sensors[0]), reward, type(reward), type(reward[0])

    tr_world.world_pub.publish(sensors, reward)
    tr_world.spin()