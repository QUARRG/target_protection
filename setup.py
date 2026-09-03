from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'target_protection'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'msg'), glob('msg/*.msg')),
    ],
    install_requires=['setuptools','numpy-quaternion','numpy'],
    zip_safe=True,
    maintainer='Dimitria Silveria',
    maintainer_email='dimitriasilveria.ds@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'agents_order = target_protection.agents_order:main',
            'gps_scanner_ii = target_protection.gps_scanner_ii:main',
            'sim_pose_bridge = target_protection.sim_pose_bridge:main',
            'sim_point_mass = target_protection.sim_point_mass:main',
            'evader = target_protection.evader:main',
            'pipeline_complete = target_protection.pipeline_complete:main',
            'simulation_experiment_controller = target_protection.simulation_experiment_controller:main'
        ],
    },
)
