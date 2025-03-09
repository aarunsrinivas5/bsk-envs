from setuptools import setup, find_packages

setup(
    name='bsk-envs',
    version='0.0.2',
    keywords='basilisk, hohmann, interplanetary, orbit discovery, transfers',
    url='https://github.com/aarunsrinivas5/bsk-envs.git',
    description='Gymnasium environments built on top of Basilisk',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'gymnasium'
    ]
)