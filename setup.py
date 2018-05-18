#! /usr/bin/env python
# System imports
from setuptools import setup, Extension, find_packages
import os
import subprocess


try:
   from Cython.Distutils import build_ext
except ImportError:
   USE_CYTHON = False
else:
    USE_CYTHON = True

packages = find_packages()

# versioning
MAJOR = 0
MINOR = 0
MICRO = 14 
ISRELEASED = False
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)


info = {'name': 'pylj',
                    'description': 'Simple teaching tool for classical MD simulation',
                    'author': 'Andrew R. McCluskey',
                    'author_email': 'arm61@bath.ac.uk',
                    'packages': packages,
                    'include_package_data': True,
                    'url':'http://pythoninchemistry.org/pylj',
                    'long_description':open('README.md').read(),
                    'setup_requires': ['numpy', 'matplotlib'],
                    'install_requires': ['numpy', 'matplotlib'],
                    'version': VERSION,
                    'license': 'MIT',
                    'classifiers': ['Development Status :: 3 - Alpha', 'Intended Audience :: Science/Research', 'Topic :: Scientific/Engineering', 'Topic :: Scientific/Engineering :: Chemistry', 'Topic :: Scientific/Engineering :: Physics', 'Programming Language :: Python :: 3']
                    }

####################################################################
# this is where setup starts
####################################################################
def setup_package():
    if USE_CYTHON:
        ext_modules = []
        HAS_NUMPY = True
        try:
            import numpy as np
        except:
            info['setup_requires'] = ['numpy']
            HAS_NUMPY = False
        if HAS_NUMPY:
            try:
                numpy_include = np.get_include()
            except AttributeError:
                numpy_include = np.get_numpy_include()
            _cslowstuff = Extension(name='pylj.comp',
                                    sources=['src/_ccomp.pyx', 'src/comp.cpp'],
                                    include_dirs=[numpy_include],
                                    language='c++',
                                    extra_compile_args=[],
                                    extra_link_args=['-lpthread']
                                    )
            ext_modules.append(_cslowstuff)
            info['cmdclass'] = {'build_ext': build_ext}
            info['ext_modules'] = ext_modules
            info['zip_safe'] = False
        else:
            print("Please install cython.")
            exit()
        try:
            setup(**info)
        except ValueError:
            print("")
            print("*****WARNING*****")
            print("PLEASE INSTALL A C++ COMPILER")
            print("*****************")
            print("")
            exit()
if __name__ == '__main__':
    setup_package()
