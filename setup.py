from setuptools import find_packages, setup

setup(
    name='wsp-matrix-butler',
    version="1.0",
    packages=find_packages(),
    python_requires='>=2.7',
    install_requires=[
        'pandas<0.24',
        'numpy>=1.14'
    ]
)
