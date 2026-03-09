from setuptools import setup, find_packages

setup(
    name='nvidia_radio',
    version='0.1',
    packages=find_packages(),
    py_modules=['hubconf'],
    include_package_data=True,
    zip_safe=False,
)
