from setuptools import setup, find_packages


# with open('test.txt', 'r') as file:
#     requirements = [line.split() for line in file if line.strip()]
#     print(requirements)

setup(
    name='helper',
    version="0.0.1",
    packages=find_packages(include=['helper', 'helper.*']), 
    # install_requires=['=='.join(req) for req in requirements],
)
