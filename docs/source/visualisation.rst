visualisation
=============

existing classes
----------------

pylj comes with four different methods for visualising the simulation that is taking place:

- JustCell
- RDF
- Scattering
- Interactions
- Energy

Full information about the existing classes can be found in the :doc:`sample` class documentation. 

building your own class
-----------------------

Using the inbuild tools it is straightforward to build your own custom sample class, or to have students design their own. 

Each sample class consists of at least an :code:`__init__` function and an :code:`update` function. The :code:`__init__` function should select the number of panes in the visualisation environment by calling the :code:`sample.environment(n)` function and setup the various windows using the different setup functions. The :code:`update` function then updates the panes each time it is called, this function should consist of a series of update functions related to each pane. A commented example of the JustCell sample class is shown below.  

.. code-block:: python 

    class JustCell(object):
        """The JustCell class will plot just the particle positions.
        This is a simplistic sampling class for quick visualisation. 
    
        Parameters
        ----------
        system: System
            The whole system information. 
        """
        def __init__(self):
            # First the fig and ax must be defined. These are the 
            # Matplotlib.figure.Figure object for the whole 
            # visualisation environment and the axes object (or 
            # array of axes objects) related to the individual panes. 
            fig, ax = environment(1)
    
            # This is a setup function (detailed below) for the 
            # particle positions 
            setup_cellview(ax, system)
    
            # Generally looks better with the tight_layout function 
            # called
            plt.tight_layout()
    
            # The Matplotlib.figure.Figure and axes object are set to
            # self variables for the class
            self.ax = ax
            self.fig = fig
    
        def update(self, system):
            """This updates the visualisation environment. Often this 
            can be slower than the cythonised force calculation so 
            use this wisely.
    
            Parameters
            ----------
            system: System
                The whole system information. 
            """
            # This is the update function (detailed below) for the 
            # particle positions
            update_cellview(self.ax, system)
    
            # The use of the `%matplotlib notebook' magic function in 
            # the Jupyter notebook means that the canvas must be 
            # redrawn
            self.fig.canvas.draw()

The :code:`setup_cellview` and :code:`update_cellview` functions have the following form.

.. code-block:: python

    def setup_cellview(ax, system):
        """Builds the particle position visualisation pane.

        Parameters
        ----------
        ax: Axes object
            The axes position that the pane should be placed in.
        system: System
            The whole system information.
        """
        # Assign the particles x and y coordinates to new lists
        xpos = system.particles['xposition']
        ypos = system.particles['yposition']
        
        # This simply defines the size of the particle such that
        # it is proportional to the box size
        mk = (1052.2 / (system.box_length - 0.78921) - 1.2174)

        # Plot the initial positions of the particles
        ax.plot(xpos, ypos, 'o', markersize=mk, 
                markeredgecolor='black', color='#34a5daff')
        
        # Make the box the right size and remove the ticks
        ax.set_xlim([0, system.box_length])
        ax.set_ylim([0, system.box_length])
        ax.set_xticks([])
        ax.set_yticks([])

    def update_cellview(ax, system):
        """Updates the particle positions visualisation pane.

        Parameters
        ----------
        ax: Axes object
            The axes position that the pane should be placed in.
        system: System
            The whole system information.
        """
        # Assign the particles x and y coordinates to new lists
        xpos = system.particles['xposition']
        ypos = system.particles['yposition']
        
        # The plotted data is accessed as an object in the axes 
        # object
        line = ax.lines[0]
        line.set_ydata(ypos)
        line.set_xdata(xpos)

Hopefully, it is clear how a custom enivornment could be created. Currently there are functions to setup and update the following panes:

- :code:`cellview`: the particle positions
- :code:`rdfview`: the radial distribution function
- :code:`diffview`: the scattering profile
- :code:`msdview`: the mean squared deviation against time
- :code:`pressureview`: the instantaneous pressure against time
- :code:`tempview`: the instantaneous temperature against time
- :code:`forceview`: the total force against time
- :code:`energyview`: the total energy against time

For those plotted against time, the samples are stored as np.arrays in the System object. To design a new sampling pane based on a different variable it may be necessary to impliment this in the System class, and the sampling of it would be added to the sample function in the particular module being used e.g. :code:`md`. 


