from setuptools import setup, find_packages

setup(
    name='nvidia_radio',
    version='0.1',
    package_dir={'nvidia_radio': '.'},
    packages=['nvidia_radio'] + ['nvidia_radio.' + p for p in find_packages()],
    include_package_data=True,
    zip_safe=False,
)
