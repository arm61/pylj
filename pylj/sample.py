import numpy as np
import matplotlib.pyplot as plt


class Scattering(object): #pragma: no cover
    """The Scattering class will plot the particle positions, radial distribution function, mean squared deviation and
    scattering profile (as a fft of the rdf). This sampling class is ideal for observing the phase transitions between
    solid, liquid, gas.

    Parameters
    ----------
    system: System
        The whole system information. 
    """
    def __init__(self, system):
        fig, ax = environment(4) 
        self.average_rdf = []
        self.r = []
        self.average_diff = []
        self.q = []
        self.initial_pos = [system.particles['xposition'], system.particles['yposition']]

        setup_cellview(ax[0, 0], system)
        setup_rdfview(ax[0, 1], system)
        setup_diffview(ax[1, 1])
        setup_msdview(ax[1, 0])

        plt.tight_layout()
        self.ax = ax
        self.fig = fig

    def update(self, system):
        """This updates the visualisation environment. Often this can be slower than the cythonised force calculation
        so used is wisely.

        Parameters
        ----------
        system: System
            The whole system information.
        """
        update_cellview(self.ax[0, 0], system)
        update_rdfview(self.ax[0, 1], system, self.average_rdf, self.r)
        update_diffview(self.ax[1, 1], system, self.average_diff, self.q)
        update_msdview(self.ax[1, 0], system)
        self.fig.canvas.draw()

    def average(self):
        gr = np.average(self.average_rdf, axis=0)
        x = np.average(self.r, axis=0)
        line = self.ax[0, 1].lines[0]
        line.set_xdata(x)
        line.set_ydata(gr)
        self.ax[0, 1].set_ylim([0, np.amax(gr) + 0.5])
        self.fig.canvas.draw()
 
        iq = np.average(self.average_diff, axis=0)
        x = np.average(self.q, axis=0)
        line = self.ax[1, 1].lines[0]
        line.set_ydata(iq)
        line.set_xdata(x)
        self.ax[1, 1].set_ylim([0, np.amax(iq) + np.amax(iq) * 0.05])
        self.ax[1, 1].set_xlim([0, np.amax(x)])


class Interactions(object): #pragma: no cover
    """The Interactions class will plot the particle positions, total force, simulation pressure and temperature. This
    class is perfect for showing the interactions between the particles and therefore the behaviour of ideal gases and
    deviation when the conditions of an ideal gas are not met.

    Parameters
    ----------
    system: System
        The whole system information.
    """
    def __init__(self, system):
        fig, ax = environment(4)

        setup_cellview(ax[0, 0], system)
        setup_forceview(ax[1, 1])
        setup_pressureview(ax[1, 0])
        setup_tempview(ax[0, 1])

        plt.tight_layout()
        self.ax = ax
        self.fig = fig

    def update(self, system):
        """This updates the visualisation environment. Often this can be slower than the cythonised force calculation
        so used is wisely.

        Parameters
        ----------
        system: System
            The whole system information.
        """
        update_cellview(self.ax[0, 0], system)
        update_forceview(self.ax[1, 1], system)
        update_tempview(self.ax[0, 1], system)
        update_pressureview(self.ax[1, 0], system)

        self.fig.canvas.draw()


class JustCell(object): #pragma: no cover
    """The JustCell class will plot just the particles positions. This is a simplistic sampling class for quick
    visualisation.

    Parameters
    ----------
    system: System
        The whole system information.
    """
    def __init__(self, system):
        fig, ax = environment(1)

        setup_cellview(ax, system)

        plt.tight_layout()

        self.ax = ax
        self.fig = fig

    def update(self, system):
        """This updates the visualisation environment. Often this can be slower than the cythonised force calculation
        so use this wisely.

        Parameters
        ----------
        system: System
            The whole system information.
        """
        update_cellview(self.ax, system)

        self.fig.canvas.draw()


class Energy(object): #pragma: no cover
    """The RDF class will plot the particle positions and radial distribution function. This sampling class is can be
    used to show the relative RDFs for solid, liquid, gas.

    Parameters
    ----------
    system: System
        The whole system information.
    """
    def __init__(self, system):
        fig, ax = environment(2)

        setup_cellview(ax[0], system)
        setup_energyview(ax[1])

        plt.tight_layout()
        self.ax = ax
        self.fig = fig

    def update(self, system):
        """This updates the visualisation environment. Often this can be slower than the cythonised force calculation
        so used is wisely.

        Parameters
        ----------
        system: System
            The whole system information.
        """
        update_cellview(self.ax[0], system)
        update_energyview(self.ax[1], system)
        self.fig.canvas.draw()


class RDF(object): #pragma: no cover
    """The RDF class will plot the particle positions and radial distribution function. This sampling class is can be
    used to show the relative RDFs for solid, liquid, gas.

    Parameters
    ----------
    system: System
        The whole system information.
    """
    def __init__(self, system):
        fig, ax = environment(2)
        self.average_rdf = []
        self.r = []
        self.average_diff = []
        self.q = []
        self.initial_pos = [system.particles['xposition'], system.particles['yposition']]

        setup_cellview(ax[0], system)
        setup_rdfview(ax[1], system)

        plt.tight_layout()
        self.ax = ax
        self.fig = fig

    def update(self, system):
        """This updates the visualisation environment. Often this can be slower than the cythonised force calculation
        so used is wisely.

        Parameters
        ----------
        system: System
            The whole system information.
        """
        update_cellview(self.ax[0], system)
        update_rdfview(self.ax[1], system, self.average_rdf, self.r)
        self.fig.canvas.draw()

    def average(self):
        gr = np.average(self.average_rdf, axis=0)
        x = np.average(self.r, axis=0)
        line = self.ax[1].lines[0]
        line.set_xdata(x)
        line.set_ydata(gr)
        self.ax[1].set_ylim([0, np.amax(gr) + np.amax(gr) * 0.05])
        self.fig.canvas.draw()


def environment(panes): #pragma: no cover
    """The visualisation environment consists of a series of panes (1, 2, or 4 are allowed). This function allows the
    number of panes in the visualisation to be defined.

    Parameters
    ----------
    panes: int
        Number of visualisation panes.

    Returns
    -------
    Matplotlib.figure.Figure object:
        The relevant Matplotlib figure.
    Axes object or array of axes objects:
        The axes related to each of the panes. For panes=1 this is a single object, for panes=2 it is a 1-D array and
        for panes=4 it is a 2-D array.
    """
    if panes == 1:
        fig, ax = plt.subplots(figsize=(4, 4))
    elif panes == 2:
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    elif panes == 4:
        fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    else:
        AttributeError('The only options for the number of panes are 1, 2, or 4')
    return fig, ax
        

def setup_cellview(ax, system): #pragma: no cover
    """Builds the particle position visualisation pane.

    Parameters
    ----------
    ax: Axes object
        The axes position that the pane should be placed in.
    system: System
        The whole system information.
    """
    xpos = system.particles['xposition']
    ypos = system.particles['yposition']
    mk = (6.00555e-8 / (system.box_length - 2.2727e-10) - 1e-10)
    ax.plot(xpos, ypos, 'o', markersize=mk, markeredgecolor='black', color='#34a5daff')
    ax.set_xlim([0, system.box_length])
    ax.set_ylim([0, system.box_length])
    ax.set_xticks([])
    ax.set_yticks([])


def setup_forceview(ax): #pragma: no cover
    """Builds the total force visualisation pane.

    Parameters
    ----------
    ax: Axes object
        The axes position that the pane should be placed in.
    """
    ax.plot([0], color='#34a5daff')
    ax.set_ylabel('Force/N', fontsize=16)
    ax.set_xlabel('Time/s', fontsize=16)


def setup_energyview(ax): #pragma: no cover
    """Builds the total force visualisation pane.

    Parameters
    ----------
    ax: Axes object
        The axes position that the pane should be placed in.
    """
    ax.plot([0], color='#34a5daff')
    ax.set_ylabel('Potential Energy/J', fontsize=16)
    ax.set_xlabel('Step', fontsize=16)


def setup_rdfview(ax, system): #pragma: no cover
    """Builds the radial distribution function visualisation pane.

    Parameters
    ----------
    ax: Axes object
        The axes position that the pane should be placed in.
    system: System
        The whole system information.
    """
    ax.plot([0], color='#34a5daff')
    ax.set_xlim([0, system.box_length / 2])
    ax.set_yticks([])
    ax.set_ylabel('RDF', fontsize=16)
    ax.set_xlabel('r/m', fontsize=16)


def setup_diffview(ax): #pragma: no cover
    """Builds the scattering profile visualisation pane.

    Parameters
    ----------
    ax: Axes object
        The axes position that the pane should be placed in.
    """
    ax.plot([0], color='#34a5daff')
    ax.set_yticks([])
    ax.set_xscale('log')
    ax.set_yscale('symlog')
    ax.set_ylabel('I(q)', fontsize=16)
    ax.set_xlabel('q/m$^{-1}$', fontsize=16)


def setup_pressureview(ax): #pragma: no cover
    """Builds the simulation instantaneous pressure visualisation pane.

    Parameters
    ----------
    ax: Axes object
        The axes position that the pane should be placed in.
    """
    ax.plot([0], color='#34a5daff')
    ax.set_ylabel(r'Pressure/$\times10^6$Pa m$^{-1}$', fontsize=16)
    ax.set_xlabel('Time/s', fontsize=16)


def setup_tempview(ax): #pragma: no cover
    """Builds the simulation instantaneous temperature visualisation pane.

    Parameters
    ----------
    ax: Axes object
        The axes position that the pane should be placed in.
    """
    ax.plot([0], color='#34a5daff')
    ax.set_ylabel('Temperature/K', fontsize=16)
    ax.set_xlabel('Time/s', fontsize=16)


def update_cellview(ax, system): #pragma: no cover
    """Updates the particle positions visualisation pane.

    Parameters
    ----------
    ax: Axes object
        The axes position that the pane should be placed in.
    system: System
        The whole system information.
    """
    x3 = system.particles['xposition']
    y3 = system.particles['yposition']
    line = ax.lines[0]
    line.set_ydata(y3)
    line.set_xdata(x3)


def update_rdfview(ax, system, average_rdf, r): #pragma: no cover
    """Updates the radial distribution function visualisation pane.

    Parameters
    ----------
    ax: Axes object
        The axes position that the pane should be placed in.
    system: System
        The whole system information.
    average_rdf: array_like
        The radial distribution functions g(r) for each timestep, to later be averaged.
    r: array_like
        The radial distribution functions r for each timestep, to later be averaged.
    """
    hist, bin_edges = np.histogram(system.distances, bins=np.linspace(0, system.box_length/2 + 0.5e-10, 100))
    gr = hist / (system.number_of_particles * (system.number_of_particles / system.box_length ** 2) * np.pi *
                 (bin_edges[:-1] + 0.5e-10 / 2.) * 0.5)
    average_rdf.append(gr)
    x = bin_edges[:-1] + 0.5e-10 / 2
    r.append(x)

    line = ax.lines[0]
    line.set_xdata(x)
    line.set_ydata(gr)
    ax.set_ylim([0, np.amax(gr) + np.amax(gr) * 0.05])


def update_diffview(ax, system, average_diff, q): #pragma: no cover
    """Updates the scattering profile visualisation pane.

    Parameters
    ----------
    ax: Axes object
        The axes position that the pane should be placed in.
    system: System
        The whole system information.
    average_diff: array_like
        The scattering profile's i(q) for each timestep, to later be averaged.
    q: array_like
        The scattering profile's q for each timestep, to later be averaged.
    """
    qw = np.logspace(np.log10(2 * np.pi / system.box_length), 10.47, num=1000, base=10)
    i = np.zeros_like(qw)
    for j in range(0, len(qw)):
        i[j] = np.sum(3.644 * (np.sin(qw[j] * system.distances))/(qw[j] * system.distances))
        if i[j] < 0:
            i[j] = 0
    x2 = qw
    y2 = i
    average_diff.append(y2)
    q.append(x2)
    line1 = ax.lines[0]
    line1.set_xdata(x2)
    line1.set_ydata(y2)
    ax.set_ylim([0, np.amax(y2) + np.amax(y2) * 0.05])
    ax.set_xlim(np.amin(x2), np.amax(x2))


def update_forceview(ax, system): #pragma: no cover
    """Updates the total force visualisation pane.

    Parameters
    ----------
    ax: Axes object
        The axes position that the pane should be placed in.
    system: System
        The whole system information.
    """
    line = ax.lines[0]
    line.set_ydata(system.force_sample)
    line.set_xdata(np.arange(0, system.step) * system.timestep_length)
    ax.set_xlim(0, system.step * system.timestep_length) 
    ax.set_ylim(np.amin(system.force_sample)-np.amax(system.force_sample) * 0.05,
                np.amax(system.force_sample)+np.amax(system.force_sample) * 0.05)


def update_energyview(ax, system): #pragma: no cover
    """Updates the total force visualisation pane.

    Parameters
    ----------
    ax: Axes object
        The axes position that the pane should be placed in.
    system: System
        The whole system information.
    """
    line = ax.lines[0]
    line.set_ydata(system.energy_sample)
    if system.force_sample != []:
        line.set_xdata(np.arange(0, system.step) * system.timestep_length)
        ax.set_xlim(0, system.step * system.timestep_length)
        ax.set_xlabel('Time/s', fontsize=16)
    else:
        line.set_xdata(np.arange(0, system.step+1))
        ax.set_xlim(0, system.step)
    ax.set_ylim(np.amin(system.energy_sample)-np.abs(np.amax(system.energy_sample)) * 0.05,
                np.amax(system.energy_sample)+np.abs(np.amax(system.energy_sample)) * 0.05)


def update_tempview(ax, system): #pragma: no cover
    """Updates the simulation instantaneous temperature visualisation pane.

    Parameters
    ----------
    ax: Axes object
        The axes position that the pane should be placed in.
    system: System
        The whole system information.
    """
    line = ax.lines[0]
    line.set_ydata(system.temperature_sample)
    line.set_xdata(np.arange(0, system.step) * system.timestep_length)
    ax.set_xlim(0, system.step * system.timestep_length) 
    ax.set_ylim(np.amin(system.temperature_sample)-np.amax(system.temperature_sample) * 0.05,
                np.amax(system.temperature_sample)+np.amax(system.temperature_sample) * 0.05)


def update_pressureview(ax, system): #pragma: no cover
    """Updates the simulation instantaneous pressure visualisation pane.

    Parameters
    ----------
    ax: Axes object
        The axes position that the pane should be placed in.
    system: System
        The whole system information.
    """
    line = ax.lines[0]
    data = system.pressure_sample * 1e6
    line.set_ydata(data)
    line.set_xdata(np.arange(0, system.step) * system.timestep_length)
    ax.set_xlim(0, system.step * system.timestep_length)
    ax.set_ylim(np.amin(data) - np.amax(data) * 0.05,
                np.amax(data) + np.amax(data) * 0.05)


def setup_msdview(ax): #pragma: no cover
    """Builds the simulation mean squared deviation visualisation pane.

    Parameters
    ----------
    ax: Axes object
        The axes position that the pane should be placed in.
    """
    ax.plot([0], color='#34a5daff')
    ax.set_ylabel('MSD/m$^2$', fontsize=16)
    ax.set_xlabel('Time/s', fontsize=16)


def update_msdview(ax, system): #pragma: no cover
    """Updates the simulation mean squared deviation visualisation pane.

    Parameters
    ----------
    ax: Axes object
        The axes position that the pane should be placed in.
    system: System
        The whole system information.
    """
    line = ax.lines[0]

    line.set_ydata(system.msd_sample)
    line.set_xdata(np.arange(0, system.step) * system.timestep_length)
    ax.set_xlim(0, system.step * system.timestep_length)
    ax.set_ylim(0, np.amax(system.msd_sample)+np.amax(system.msd_sample) * 0.05)
