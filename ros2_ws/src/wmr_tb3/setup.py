import os
from glob import glob
from setuptools import setup

package_name = 'wmr_tb3'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'models', 'tb3_burger_slip'),
         glob('models/tb3_burger_slip/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='eugene',
    maintainer_email='n28141050@gs.ncku.edu.tw',
    description='Slipping trajectory tracking control on TurtleBot3.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tracking_node = wmr_tb3.tracking_node:main',
            'slip_injector = wmr_tb3.slip_injector:main',
        ],
    },
)
