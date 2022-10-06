#!/usr/bin/python

from setuptools import setup, find_packages

setup(
    name   ='PeakDetector',
    version='0.1.0',
    author='Fernando Pujaico Rivera',
    author_email='fernando.pujaico.rivera@gmail.com',
    packages=['PeakDetector'],
    #scripts=['bin/script1','bin/script2'],
    url='https://github.com/trucomanx/PeakDetector',
    license='GPLv3',
    description='Creates a peak detector',
    #long_description=open('README.txt').read(),
    install_requires=[
       "numpy" #"Django >= 1.1.1",
    ],
)

#! python setup.py sdist bdist_wheel
# Upload to PyPi
# or 
#! pip3 install dist/PeakDetector-0.1.tar.gz 
