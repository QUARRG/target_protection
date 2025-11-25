from setuptools import find_packages, setup
import os
from glob import glob
package_name = 'crazy_encirclement'

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
    ],
    install_requires=['setuptools','numpy-quaternion','numpy'],
    zip_safe=True,
    maintainer='Dimitria Silveria',
    maintainer_email='dimitriasilveria.ds@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'encirclement = crazy_encirclement.encirclement_node:main',
            'crazy_circle = crazy_encirclement.crazy_circle:main',
            'agents_order = crazy_encirclement.agents_order:main',
            'circle_distortion_filter_gps = crazy_encirclement.circle_distortion_filter_gps:main',
            'circle_distortion_baseline = crazy_encirclement.circle_distortion_baseline:main',
            'circle_distortion_filter_relative = crazy_encirclement.circle_distortion_filter_relative:main',
            'gps = crazy_encirclement.gps:main',
            'gps_scanner = crazy_encirclement.gps_scanner:main',
        ],
    },
)
