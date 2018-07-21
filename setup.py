from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['tensorflow==1.8', 'setuptools>=34.0.0', 'requests<3.0.0dev,>=2.18.0', 'six==1.11.0', 'numpy==1.14.2']

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='valueprediction trainer package.'
)
