from setuptools import setup, find_packages

setup(
    name='bsk-envs',
    version='0.0.2',
    keywords='basilisk, hohmann, interplanetary, transfers',
    url='https://github.com/aarunsrinivas5/bsk-transfers.git',
    description='Gymnasium environments built on top of Basilisk for simulating transfers',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'gymnasium'
    ]
)