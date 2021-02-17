#! /usr/bin/env python
# System imports
from setuptools import setup, Extension, find_packages
from os import path
import io

packages = find_packages()

# versioning
MAJOR = 1
MINOR = 4
MICRO = 1
ISRELEASED = True
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)


this_directory = path.abspath(path.dirname(__file__))
with io.open(path.join(this_directory, 'README.md')) as f:
    long_description = f.read()

info = {
        'name': 'pylj',
        'description': 'Simple teaching tool for classical MD simulation',
        'author': 'Andrew R. McCluskey',
        'author_email': 'arm61@bath.ac.uk',
        'packages': packages,
        'include_package_data': True,
        'setup_requires': ['jupyter', 'numpy', 'matplotlib', 'cython', 'numba'],
        'install_requires': ['jupyter', 'numpy', 'matplotlib', 'cython', 'numba'],
        'version': VERSION,
        'license': 'MIT',
        'long_description': long_description,
        'long_description_content_type': 'text/markdown',
        'classifiers': ['Development Status :: 5 - Production/Stable',
                        'Framework :: Jupyter',
                        'Intended Audience :: Science/Research',
                        'License :: OSI Approved :: MIT License',
                        'Natural Language :: English',
                        'Operating System :: OS Independent',
                        'Programming Language :: Python :: 2.7',
                        'Programming Language :: Python :: 3.5',
                        'Programming Language :: Python :: 3.6',
                        'Programming Language :: Python :: 3.7',
                        'Topic :: Scientific/Engineering',
                        'Topic :: Scientific/Engineering :: Chemistry',
                        'Topic :: Scientific/Engineering :: Physics']
        }

####################################################################
# this is where setup starts
####################################################################


def setup_package():
    setup(**info)


if __name__ == '__main__':
    setup_package()
