from setuptools import setup, find_packages

setup(
    name='pyvacy',
    version='0.0.1',
    description='Privacy preserving deep learning for PyTorch',
    author='Chris Waites',
    author_email='cwaites10@gmail.com',

    license='Apache 2.0',
    url='http://github.com/ChrisWaites/pyvacy',
    packages=find_packages(),
    install_requires=['pytorch', 'torch-vision'],
)
