from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'visual_navigation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

         # Install configs
        (os.path.join('share', package_name, 'configs'), glob('configs/*.yaml')),

        # Install RViz configs
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),

        # Install launch files.
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Hardik Shah',
    maintainer_email='hardikns@jpl.nasa.gov',
    description='Long-range visual navigation and object search.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lrn = visual_navigation.lrn.nav:main',
            'gps_viz = visual_navigation.gps.gps_viz:main',
            'gps_save = visual_navigation.gps.save_gps_path:main',
            'metrics_save = visual_navigation.gps.save_metrics:main',
            'img_frontier_nav = visual_navigation.imgfrontier_nav.nav:main',
            'geo_frontier_nav = visual_navigation.geofrontier_nav.nav:main',
            'wildos = visual_navigation.wildos.nav:main',
            'obj_mask_triangulation = visual_navigation.explorfm_triangulation.obj_mask_triangulation:main',
            'viz_net = visual_navigation.imgfrontier_nav.viz_net:main',
            'explorfm_triangulate = visual_navigation.explorfm_triangulation.explorfm_triangulator:main',
        ],
    },
)
