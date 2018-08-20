from setuptools import setup, find_packages
import subprocess

setup(
    name="hxtools",
    packages=find_packages(),
    author = "Gabriel Rocklin",
    author_email = "grocklin@gmail.com",
    package_data={},
    install_requires=[
        l.strip() for l in open("requirements.txt").readlines()],
)
