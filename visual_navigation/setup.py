from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'img_vlms'

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
            'img_sub = img_vlms.vlm_viz.img_sub:main',
            'img_sam = img_vlms.vlm_viz.img_sam:main',
            'img_dinov2_pca = img_vlms.vlm_viz.img_dinofeat:main',
            'img_ram_gdino = img_vlms.vlm_viz.img_ram_gdino:main',
            'img_ramp_gdino = img_vlms.vlm_viz.img_ramp_gdino:main',
            'img_radio = img_vlms.vlm_viz.img_radio:main',
            'radio_triangulate = img_vlms.radio_triangulation.demo_radio_triangulator:main',
            'debug_tf = img_vlms.radio_triangulation.debug_tf:main',
            'lrn = img_vlms.lrn.nav:main',
            'gps_viz = img_vlms.gps.gps_viz:main',
            'gps_save = img_vlms.gps.save_gps_path:main',
            'metrics_save = img_vlms.gps.save_metrics:main',
            'img_frontier_nav = img_vlms.imgfrontier_nav.nav:main',
            'geo_frontier_nav = img_vlms.geofrontier_nav.nav:main',
            'goalagnostic_scoring = img_vlms.goalagnostic_geofrontier_nav.nav:main',
            'obj_mask_triangulation = img_vlms.radio_triangulation.obj_mask_triangulation:main',
            'viz_net = img_vlms.imgfrontier_nav.viz_net:main',
            'pcl_colorizer = img_vlms.imgfrontier_nav.pcl_color:main',
            'map_colorizer = img_vlms.imgfrontier_nav.map_color:main',
        ],
    },
)
