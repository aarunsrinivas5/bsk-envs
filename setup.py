from setuptools import setup

setup(
    name='bsk-transfers',
    version='0.0.2',
    keywords='basilisk, hohmann, interplanetary, transfer',
    url='https://github.com/aarunsrinivas5/bsk-transfers.git',
    description='Gymnasium environments built on top of Basilisk for simulating transfers',
    packages=['bsk-transfers'],
    install_requires=[
        'numpy',
        'gymnasium'
    ]
)