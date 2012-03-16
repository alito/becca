#!/usr/bin/env python

from distutils.core import setup

setup(name='Becca',
	  version='0.4.0',
      description='BECCA is a general learning program for use in any robot or embodied system',
      author='Brandon Rohrer',
      author_email='brohrer@gmail.com',
      url='http://www.sandia.gov/~brrohre/becca.html',
      packages=['becca', 'becca.agent', 'becca.experiment', 'becca.worlds'],
     )
