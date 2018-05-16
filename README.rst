pylj
====

.. image:: https://github.com/arm61/pylj/blob/master/logo/logo.png?raw=true

.. image:: https://zenodo.org/badge/119863480.svg
   :target: https://zenodo.org/badge/latestdoi/119863480
.. image:: https://readthedocs.org/projects/pylj/badge/?version=latest
   :target: http://pylj.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

how to cite pylj
----------------
Thank you for using pylj. If you use this code in a teaching laboratory or a publication we would greatly appreciate if you would use the following citation.
Andrew R. McCluskey, Benjamin J. Morgan, Karen J. Edler, Stephen C. Parker (2018). pylj, version 0.0.6a. Released: 2018-05-15, DOI: 10.5281/zenodo.1212792. 

what is pylj?
-------------

pylj is an open-source library to facilitate student interaction with classical simulation. It is designed to operate within the Jupyter notebook framework, making it easy to implement in the classroom, or computer lab. Additionally, due to the open-source, and documented, nature of the code it is easy for educators to add unique, custom extensions. 

what does pylj offer?
---------------------

Currently pylj will perform the simulation of a 2D argon system by molecular dynamics, with both NVE and NVT ensembles available and making use of a Velocity-Verlet integrator. A series of sampling classes exist (found in the sample module), such as the Interactions and Scattering classes. However, it is straightforward to build a custom sampling class either from scratch or using the sampling class building tools. 

example exercises
-----------------

We are currently in the process of developing example laboratory exercises that make use of pylj. These will include a study of ideal and non-ideal gas conditions and the effect of the phase transitions on the radial distribution function, scattering profiles and mean squared deviation. 

how do i get pylj?
------------------

If you are interested in using pylj, in any sense, fork the code at http://www.github.com/arm61/pylj or email Andrew (arm61 'at' bath.ac.uk). We are currently investigating the feasibility of hosting a freely available test instance on Amazon Web Services.

requirements
------------
To run pylj locally it is necessary to have:

- python 3
- numpy
- matplotlib
- cython
- C++ complier

todo
----
- get a cite funciton 
- unit testing 
- complete example lesssons
- add Monte-Carlo
- add energy minimisation
