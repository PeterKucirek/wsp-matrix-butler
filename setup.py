from setuptools import find_packages, setup

setup(
    name='wsp-matrix-butler',
    version='1.1.0',
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
