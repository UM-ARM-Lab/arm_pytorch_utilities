from setuptools import setup

setup(
    name='arm_pytorch_utilities',
    version='0.2.0',
    packages=['arm_pytorch_utilities'],
    url='https://github.com/UM-ARM-Lab/arm_pytorch_utilities',
    license='MIT',
    author='zhsh',
    author_email='zhsh@umich.edu',
    description='Utilities for working with pytorch',
    install_requires=[
        'torch',
        'numpy',
        'matplotlib', 'scipy'
    ]
)
