from setuptools import setup, find_packages

setup(
    name='explorfm',
    version='0.1.0',
    description='ExploRFM: Downstream heads and utilities for WildOS',
    packages=find_packages(),
    install_requires=[
        'nvidia_radio',
    ],
    include_package_data=True,
    zip_safe=False,
)