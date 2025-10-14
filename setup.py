from setuptools import setup, find_packages

setup(
    name='space_rover',
    version='0.1',
    packages=find_packages(exclude=['tests', 'docs']),
    install_requires=[],
)
