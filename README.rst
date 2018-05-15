pylj
====

.. image:: https://github.com/arm61/pylj/blob/master/logo/logo.png?raw=true
   :height: 100 px
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
- build webpages
- unit testing 
- complete example lesssons
- add Monte-Carlo
- add energy minimisation
