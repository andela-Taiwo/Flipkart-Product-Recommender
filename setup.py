from setuptools import setup,find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="FLIPKART RECOMMENDER SYSTEM",
    version="0.1",
    author="Taiwo Sokunbi",
    packages=find_packages(),
    install_requires = requirements,
)