from os import path
from pkg_resources import safe_version
from setuptools import find_packages, setup

version = {}
with open(path.join(path.dirname(path.realpath(__file__)), 'matrix_butler', 'version.py')) as fp:
    exec(fp.read(), {}, version)
version_string = safe_version(version['__version__'])

setup(
    name='wsp-matrix-butler',
    version=version_string,
    description='A SQLite-based mini-file system for organizing binary files for the Greater Golden Horseshoe Model',
    url='https://github.com/wsp-sag/wsp-matrix-butler',
    author='WSP',
    maintatiner='Brian Cheung',
    maintainer_email='brian.cheung@wsp.com',
    classifiers=[
        'License :: OSI Approved :: MIT License'
    ],
    packages=find_packages(),
    install_requires=[
        'pandas>=0.16',
        'numpy>=1.14'
    ],
    python_requires='>=2.7'
)
