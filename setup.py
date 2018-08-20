from setuptools import setup, find_packages
import subprocess

setup(
    name="hxtools",
    version = "0.0.1",
    packages=find_packages("src"),
    package_dir={"": "src"},
    author = "Gabriel Rocklin",
    author_email = "grocklin@gmail.com",
    package_data={},
    install_requires=[
        l.strip() for l in open("requirements.txt").readlines()],
)
