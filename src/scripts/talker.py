#!/usr/bin/env python
import rospy
from std_msgs.msg import String

def callback(data):
    rospy.loginfo(rospy.get_name() + ": I heard %s" % data.data)

def talker():
    pub = rospy.Publisher('chatter', String)
    rospy.init_node('talker')
    rospy.Subscriber("chatter2", String, callback)
    while not rospy.is_shutdown():
        str = "hello world %s" % rospy.get_time()
        rospy.loginfo(str)
        pub.publish(String(str))
        rospy.sleep(1.0)

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
