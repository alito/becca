##BECCA

BECCA is a general learning program for use in any robot or embodied system. It consists of an automatic feature creator and a model-based reinforcement learner. When using BECCA, a robot learns to do whatever it is rewarded to do, and continues learning throughout its lifetime.

Get the current release, BECCA 0.4.5. It is free and open source. Just download (3MB), unzip, and run with Python. See the included README and Users Guide for documentation. The Google group BECCA_users provides a forum for users to share questions, solutions, and experiences. Additional forums, tools, and documentation are available at openbecca.org, the homepage for the open source BECCA project.

Join the BECCA user Google group

BECCA is a computer program, a robot brain that can learn from its experiences and create abstract concepts, in the roughly same way that children do. I hope that one day it will help robots do everything I can do, including talk with people, climb mountains, clean the kitchen, and build other robots. But I'll be happy if it helps a robot to be as smart as a dog.

I have chosen to cast the challenge of natural world interaction as a reinforcement learning (RL) problem: an agent takes actions and receives sensory information at each time step. Its goal is to maximize its reward. The problem formulation assumes nothing else about the nature or structure of the environment.

In order to address this general RL problem, I develop biologically-motivated algorithms and incorporate them into a Brain-Emulating Cognition and Control Architecture (BECCA). Although it is under continual development, a recent incarnation of BECCA is diagrammed below. A more complete technical description can be found in the users guide in this repo, in Appendices A-C.
