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
        (os.path.join('share', package_name, 'msg'), glob('msg/*.msg')),
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
            'agents_order = crazy_encirclement.agents_order:main',
            'gps_scanner_ii = crazy_encirclement.gps_scanner_ii:main',
            'evader = crazy_encirclement.evader:main',
            'pipeline_complete = crazy_encirclement.pipeline_complete:main'
        ],
    },
)
