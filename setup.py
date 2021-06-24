from setuptools import find_packages, setup

import versioneer

setup(
    name='wsp-matrix-butler',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description='A SQLite-based mini-file system for organizing binary files for the Greater Golden Horseshoe Model',
    url='https://github.com/wsp-sag/wsp-matrix-butler',
    author='WSP',
    maintainer='Brian Cheung',
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
