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
        (os.path.join('share', package_name, 'third_party', 'Grounded-Segment-Anything', 'checkpoints'),
         glob('third_party/Grounded-Segment-Anything/checkpoints/*')),
         (os.path.join('share', package_name, 'third_party', 'Grounded-Segment-Anything', 'configs'),
         glob('third_party/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/*')),
        (os.path.join('share', package_name, 'third_party', 'Grounded-SAM-2', 'checkpoints'),
         glob('third_party/Grounded-SAM-2/checkpoints/*')),
        (os.path.join('share', package_name, 'third_party', 'Grounded-SAM-2', 'configs'),
         glob('third_party/Grounded-SAM-2/sam2/configs/sam2.1/*')),

         # Install configs
        (os.path.join('share', package_name, 'configs'), glob('configs/*.yaml')),

        # Install RViz configs
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),

        # Install launch files.
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='scarecrow',
    maintainer_email='hardikns@jpl.nasa.gov',
    description='Run VLMs on the robot camera feed.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'img_sub = visual_navigation.vlm_viz.img_sub:main',
            'img_sam = visual_navigation.vlm_viz.img_sam:main',
            'img_dinov2_pca = visual_navigation.vlm_viz.img_dinofeat:main',
            'img_ram_gdino = visual_navigation.vlm_viz.img_ram_gdino:main',
            'img_ramp_gdino = visual_navigation.vlm_viz.img_ramp_gdino:main',
            'img_radio = visual_navigation.vlm_viz.img_radio:main',
            'radio_triangulate = visual_navigation.radio_triangulation.demo_radio_triangulator:main',
            'debug_tf = visual_navigation.radio_triangulation.debug_tf:main',
            'lrn = visual_navigation.lrn.nav:main',
            'gps_viz = visual_navigation.gps.gps_viz:main',
            'gps_save = visual_navigation.gps.save_gps_path:main',
            'metrics_save = visual_navigation.gps.save_metrics:main',
            'img_frontier_nav = visual_navigation.imgfrontier_nav.nav:main',
            'geo_frontier_nav = visual_navigation.geofrontier_nav.nav:main',
            'goalagnostic_scoring = visual_navigation.goalagnostic_geofrontier_nav.nav:main',
            'obj_mask_triangulation = visual_navigation.radio_triangulation.obj_mask_triangulation:main',
            'viz_net = visual_navigation.imgfrontier_nav.viz_net:main',
            'pcl_colorizer = visual_navigation.imgfrontier_nav.pcl_color:main',
            'map_colorizer = visual_navigation.imgfrontier_nav.map_color:main',
            'radio_triangulate = visual_navigation.radio_triangulation.radio_triangulator:main',
        ],
    },
)
