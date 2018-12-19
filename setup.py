#! /usr/bin/env python
# System imports
from setuptools import setup, Extension, find_packages
from os import path
import io


try:
    from Cython.Distutils import build_ext
except ImportError:
    USE_CYTHON = False
else:
    USE_CYTHON = True

packages = find_packages()

# versioning
MAJOR = 1
MINOR = 2
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

    if USE_CYTHON:
        # Obtain the numpy include directory.  This logic works across numpy
        # versions.
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

            # cslowstff extension module
            _cslowstuff = Extension(
                                  name='pylj.comp',
                                  sources=['src/_ccomp.pyx',
                                           'src/comp.cpp'],
                                  include_dirs=[numpy_include],
                                  language='c++',
                                  extra_compile_args=[],
                                  extra_link_args=['-lpthread']
                                  # libraries=
                                  # extra_compile_args = "...".split(),
                                  )
            ext_modules.append(_cslowstuff)

            info['cmdclass'] = {'build_ext': build_ext}
            info['ext_modules'] = ext_modules
            info['zip_safe'] = False

    try:
        setup(**info)
    except ValueError:
        # there probably wasn't a C-compiler (windows). Try removing extension
        # compilation
        print("")
        print("*****WARNING*****")
        print("Please install a C++ compiler. If installing in windows you "
              "should then install from Visual Studio command prompt (this "
              "makes C compiler available")
        print("*****************")
        print("")
        info.pop('cmdclass')
        info.pop('ext_modules')
        setup(**info)


if __name__ == '__main__':
    setup_package()
