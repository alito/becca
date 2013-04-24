#!/usr/bin/env python
import rospy
from becca_world_test.msg import world_results
from becca_world_test.msg import agent_commands
from rospy.numpy_msg import numpy_msg


import numpy as np
from agent.agent import Agent 

        
class test_ros_agent:
    def __init__(self, restore=False, show=True, agent_name="test_agent"):
        self.agent_pub = rospy.Publisher('agent_commands', numpy_msg(agent_commands))
        rospy.init_node('agent1')
        rospy.Subscriber("world_results", numpy_msg(world_results), self.callback)
        self.agent = Agent(9, 9, 18, agent_name=agent_name, show=show)        
        #from world.set_agent_parameters
        self.agent.reward_min = -100.
        self.agent.reward_max = 100.
        
        if restore:
            agent=agent.restore()
        
    def spin(self):
        print "spinning"
        rospy.spin()
    
    def step(self, sensors, reward):
        return self.agent.step(sensors, reward)
   
    def callback(self, data):
       print "got ", data
       actions = self.step(data.sensors, data.reward)
       actions = [x[0] for x in actions]
       print actions, type(actions), type(actions[0])
       self.agent_pub.publish(np.array(actions))
      
if __name__ == '__main__':
    try:
        global t_agent
        t_agent = test_ros_agent()
        t_agent.spin()
            
    except rospy.ROSInterruptException:
        pass