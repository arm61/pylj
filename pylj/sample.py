import numpy as np
import matplotlib.pyplot as plt
import time

"""
These are a series of classes related to how the system should be sampled.
"""

class Scattering(object):
    """The scattering class will plot the particle positions, radial distribution function, instantaneous pressure
    and the scattering profile (determined as a fft for the rdf).

    Parameters
    ----------
    system: System
        The whole system information. 
    """
    def __init__(self, system):
        fig, ax = plt.subplots(2, 2, figsize=(9, 9))

        ax[0, 1].plot([0]*20, color='#34a5daff')
        ax[0, 1].set_xlim([0, system.box_length/2])
        ax[0, 1].set_xticks([])
        ax[0, 1].set_yticks([])
        ax[0, 1].set_ylabel('RDF', fontsize=16)
        ax[0, 1].set_xlabel('r', fontsize=16)
        self.step_text = ax[0, 1].text(0.98, 0.93, 'Time={:.1f}'.format(system.step), transform=ax[0, 1].transAxes,
                                       fontsize=12, horizontalalignment='right', verticalalignment='bottom')
        ax[1, 1].plot([0]*20, color='#34a5daff')
        ax[1, 1].set_xticks([])
        ax[1, 1].set_yticks([])
        ax[1, 1].set_ylabel('log(I(q))', fontsize=16)
        ax[1, 1].set_xlabel('q', fontsize=16)

        ax[0, 0].plot([0]*20, 'o', markersize=14, markeredgecolor='black', color='#34a5daff')
        ax[0, 0].set_xlim([0, system.box_length])
        ax[0, 0].set_ylim([0, system.box_length])
        ax[0, 0].set_xticks([])
        ax[0, 0].set_yticks([])

        ax[1, 0].plot([0] * 20, color='#34a5daff')
        ax[1, 0].set_ylabel('Pressure', fontsize=16)
        ax[1, 0].set_xlabel('Step', fontsize=16)
        #self.temp_text = ax[1, 0].text(0.98, 0.05, 'Pressure={:f}'.format(np.average(system.press_array)),
        #                               transform=ax[1, 0].transAxes, fontsize=12, horizontalalignment='right',
        #                               verticalalignment='bottom')

        plt.tight_layout()

        self.ax = ax
        self.fig = fig
        self.avgr = []
        self.xgr = []
        self.aiq = []
        self.xiq = []

    def update(self, particles, system):
        hist, bin_edges = np.histogram(system.distances, bins=np.arange(0, 12.5, system.bin_width))
        gr = hist / (system.number_of_particles * (system.number_of_particles / system.box_length ** 2) * np.pi *
                     (bin_edges[:-1] + system.bin_width / 2.) * system.bin_width)
        x = bin_edges[:-1] + system.bin_width / 2
        self.avgr.append(gr)
        self.xgr = x

        line = self.ax[0, 1].lines[0]
        line.set_xdata(x)
        line.set_ydata(gr)
        self.ax[0, 1].set_ylim([0, np.amax(gr)+0.5])
        self.step_text.set_text('Time={:.1f}'.format(system.time))

        x2 = np.log10(np.fft.rfftfreq(len(gr))[5:])
        y2 = np.log10(np.fft.rfft(gr)[5:])
        line1 = self.ax[1, 1].lines[0]
        line1.set_xdata(x2)
        line1.set_ydata(y2)
        self.ax[1, 1].set_ylim([np.amin(y2)-np.amax(y2)*0.05, np.amax(y2)+np.amax(y2)*0.05])
        self.ax[1, 1].set_xlim([np.amin(x2), np.amax(x2)])
        self.aiq.append(y2)
        self.xiq = x2

        x3 = np.array([])
        y3 = np.array([])
        for i in range(0, particles.size):
            x3 = np.append(x3, particles[i].xpos)
            y3 = np.append(y3, particles[i].ypos)

        line2 = self.ax[0, 0].lines[0]
        line2.set_ydata(y3)
        line2.set_xdata(x3)

        line3 = self.ax[1, 0].lines[0]
        line3.set_ydata(system.press_array)
        line3.set_xdata(np.arange(0, len(system.press_array)))
        self.ax[1, 0].set_xlim(0, len(system.press_array))
        self.ax[1, 0].set_ylim(np.amin(system.press_array) - np.amax(system.press_array) * 0.05,
                               np.amax(system.press_array) + np.amax(system.press_array) * 0.05)
        #self.temp_text.set_text('Pressure={:.0f}+/-{:.0f}'.format(np.average(system.press_array[-100:]),
        #                                                        np.std(system.press_array[-100:]) / 100))


        self.fig.canvas.draw()

    def average(self):
        gr = np.average(self.avgr, axis=0)
        x = self.xgr
        line = self.ax[0, 1].lines[0]
        line.set_xdata(x)
        line.set_ydata(gr)
        self.ax[0, 1].set_ylim([0, np.amax(gr) + 0.5])
        self.step_text.set_text('Average')

        iq = np.average(self.aiq, axis=0)
        x = self.xiq
        line = self.ax[1, 1].lines[0]
        line.set_ydata(iq)
        line.set_xdata(x)
        self.ax[1, 1].set_ylim([np.amin(iq) - np.amax(iq) * 0.05, np.amax(iq) + np.amax(iq) * 0.05])
        self.ax[1, 1].set_xlim([np.amin(x), np.amax(x)])


class JustCell(object):
    def __init__(self, system):
        fig, ax = plt.subplots(figsize=(4.5, 4.5))

        ax.plot([0]*10, 'o', markersize=14, markeredgecolor='black')
        ax.set_xlim([0, system.box_length])
        ax.set_ylim([0, system.box_length])
        ax.set_xticks([])
        ax.set_yticks([])
        self.temp_text = ax.text(0.98, 0.02, '', transform=ax.transAxes, fontsize=16, horizontalalignment='right',
                                 verticalalignment='bottom')

        plt.tight_layout()

        self.ax = ax
        self.fig = fig

    def update(self, particles, system, text):
        x3 = np.array([])
        y3 = np.array([])
        for i in range(0, particles.size):
            x3 = np.append(x3, particles[i].xpos)
            y3 = np.append(y3, particles[i].ypos)

        line2 = self.ax.lines[0]
        line2.set_ydata(y3)
        line2.set_xdata(x3)
        self.temp_text.set_text(text)
        self.fig.canvas.draw()


class EnergyMinimisation(object):
    def __init__(self, system, energy):
        fig, ax = plt.subplots(1, 2, figsize=(9, 4.5))

        ax[1].plot([0] * 20)
        ax[1].set_ylabel('$Energy$', fontsize=16)
        ax[1].set_xlabel('$Step$', fontsize=16)
        self.step_text = ax[1].text(0.98, 0.95, 'Current energy={:.2e}'.format(energy), transform=ax[1].transAxes,
                                    fontsize=12, horizontalalignment='right', verticalalignment='bottom')

        ax[0].plot([0] * 20, 'o', markersize=14, markeredgecolor='black')
        ax[0].set_xlim([0, system.box_length])
        ax[0].set_ylim([0, system.box_length])
        ax[0].set_xticks([])
        ax[0].set_yticks([])

        plt.tight_layout()
        self.ax = ax
        self.fig = fig
        self.energy_array = []

    def update(self, particles, system, energy):
        self.energy_array.append(energy)

        line = self.ax[1].lines[0]
        line.set_xdata(np.arange(0, len(self.energy_array)))
        line.set_ydata(self.energy_array)
        self.ax[1].set_ylim([np.amin(self.energy_array) - 0.1 * np.amax(self.energy_array),
                             np.amax(self.energy_array) + 0.1 * np.amax(self.energy_array)])
        self.ax[1].set_xlim([0, len(self.energy_array)])
        self.step_text.set_text('Current energy={:.2e}'.format(energy))

        x3 = np.array([])
        y3 = np.array([])
        for i in range(0, particles.size):
            x3 = np.append(x3, particles[i].xpos)
            y3 = np.append(y3, particles[i].ypos)

        line2 = self.ax[0].lines[0]
        line2.set_ydata(y3)
        line2.set_xdata(x3)

        self.fig.canvas.draw()



class ShowForce(object):
    def __init__(self, system):
        fig, ax = plt.subplots(figsize=(4.5, 4.5))

        ax.plot([-1]*10, '-', color='k')
        ax.plot([-1]*10, '-', color='k')
        ax.plot([-1]*10, '-', color='k')
        ax.plot([0]*10, 'o', markersize=14, markeredgecolor='black', color='b')
        ax.set_xlim([0, system.box_length])
        ax.set_ylim([0, system.box_length])
        ax.set_xticks([])
        ax.set_yticks([])
        self.text1 = ax.text(0.98, 0.95, 'fx' + '={:.2f}'.format(0), transform=ax.transAxes,
                                 fontsize=12, horizontalalignment='right', verticalalignment='bottom')
        self.text2 = ax.text(0.98, 0.90, 'fy' + '={:.2f}'.format(0), transform=ax.transAxes,
                                 fontsize=12, horizontalalignment='right', verticalalignment='bottom')

        plt.tight_layout()

        self.ax = ax
        self.fig = fig
        self.box = system.box_length

    def update(self, particles, system):
        x3 = np.array([])
        y3 = np.array([])
        for i in range(0, particles.size):
            x3 = np.append(x3, particles[i].xpos)
            y3 = np.append(y3, particles[i].ypos)

        line2 = self.ax.lines[3]
        line2.set_ydata(y3)
        line2.set_xdata(x3)
        self.fig.canvas.draw()

    def draw_force(self, particles, i, j, xf, yf):
        x4 = [particles[i].xpos, particles[j].xpos]
        y4 = [particles[i].ypos, particles[j].ypos]
        dx = particles[i].xpos - particles[j].xpos
        dy = particles[i].ypos - particles[j].ypos
        line3 = self.ax.lines[0]
        line4 = self.ax.lines[1]
        line5 = self.ax.lines[2]
        if np.abs(dx) < self.box / 2.:
            if np.abs(dy) < self.box / 2.:
                line3.set_xdata(x4)
                line4.set_xdata(x4)
                line3.set_ydata(y4)
                y_new2 = [-1, -1]
                line4.set_ydata(y_new2)
            else:
                line3.set_xdata(x4)
                line4.set_xdata(x4)
                if y4[0] > y4[1]:
                    y_new1 = [y4[0], y4[1]+self.box]
                    y_new2 = [y4[0]-self.box, y4[1]]
                else:
                    y_new1 = [y4[0], y4[1] - self.box]
                    y_new2 = [y4[0] + self.box, y4[1]]
                line3.set_ydata(y_new1)
                line4.set_ydata(y_new2)
        else:
            if np.abs(dy) < self.box / 2.:
                if x4[0] > x4[1]:
                    x_new1 = [x4[0], x4[1]+self.box]
                    x_new2 = [x4[0]-self.box, x4[1]]
                else:
                    x_new1 = [x4[0], x4[1] - self.box]
                    x_new2 = [x4[0] + self.box, x4[1]]
                line3.set_xdata(x_new1)
                line4.set_xdata(x_new2)
                line3.set_ydata(y4)
                line4.set_ydata(y4)
            else:
                if x4[0] > x4[1]:
                    x_new1 = [x4[0], x4[1]+self.box]
                    x_new2 = [x4[0]-self.box, x4[1]]
                else:
                    x_new1 = [x4[0], x4[1] - self.box]
                    x_new2 = [x4[0] + self.box, x4[1]]
                line3.set_xdata(x_new1)
                line4.set_xdata(x_new2)
                if y4[0] > y4[1]:
                    y_new1 = [y4[0], y4[1] + self.box]
                    y_new2 = [y4[0] - self.box, y4[1]]
                else:
                    y_new1 = [y4[0], y4[1] - self.box]
                    y_new2 = [y4[0] + self.box, y4[1]]
                line3.set_ydata(y_new1)
                line4.set_ydata(y_new2)
        self.text1.set_text('fx' + '={:.2f}'.format(xf))
        self.text2.set_text('fy' + '={:.2f}'.format(yf))
        self.fig.canvas.draw()
        time.sleep(4)

    def clear_force(self):
        x5 = []
        y5 = []
        line3 = self.ax.lines[0]
        line3.set_xdata(x5)
        line3.set_ydata(y5)
        line4 = self.ax.lines[1]
        line4.set_xdata(x5)
        line4.set_ydata(y5)
        self.text1.set_text('')
        self.text2.set_text('')
        self.fig.canvas.draw()


class RDF(object):
    def __init__(self, system):
        fig, ax = plt.subplots(1, 2, figsize=(9, 4.5))

        ax[1].plot([0] * 20)
        ax[1].set_xlim([0, system.box_length / 2])
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        ax[1].set_ylabel('RDF', fontsize=16)
        ax[1].set_xlabel('r', fontsize=16)
        self.step_text = ax[1].text(0.98, 0.95, 'Time={:.1f}'.format(system.step), transform=ax[1].transAxes,
                                    fontsize=12, horizontalalignment='right', verticalalignment='bottom')

        ax[0].plot([0] * 20, 'o', markersize=14, markeredgecolor='black')
        ax[0].set_xlim([0, system.box_length])
        ax[0].set_ylim([0, system.box_length])
        ax[0].set_xticks([])
        ax[0].set_yticks([])

        plt.tight_layout()
        self.ax = ax
        self.fig = fig
        self.avgr = []
        self.xgr = []

    def update(self, particles, system):
        hist, bin_edges = np.histogram(system.distances, bins=np.arange(0, 12.5, system.bin_width))
        gr = hist / (system.number_of_particles * (system.number_of_particles / system.box_length ** 2) * np.pi *
                     (bin_edges[:-1] + system.bin_width / 2.) * system.bin_width)
        self.avgr.append(gr)
        x = bin_edges[:-1] + system.bin_width / 2
        self.xgr = x

        line = self.ax[1].lines[0]
        line.set_xdata(x)
        line.set_ydata(gr)
        self.ax[1].set_ylim([0, np.amax(gr) + 0.5])
        self.step_text.set_text('Time={:.1f}'.format(system.time))

        x3 = np.array([])
        y3 = np.array([])
        for i in range(0, particles.size):
            x3 = np.append(x3, particles[i].xpos)
            y3 = np.append(y3, particles[i].ypos)

        line2 = self.ax[0].lines[0]
        line2.set_ydata(y3)
        line2.set_xdata(x3)

        self.fig.canvas.draw()

    def average_rdf(self):
        gr = np.average(self.avgr, axis=0)
        x = self.xgr
        line = self.ax[1].lines[0]
        line.set_xdata(x)
        line.set_ydata(gr)
        self.ax[1].set_ylim([0, np.amax(gr) + 0.5])
        self.step_text.set_text('Average')

class Temperature(object):
    def __init__(self, system):
        fig, ax = plt.subplots(1, 2, figsize=(9, 4.5))

        ax[0].plot([0] * 20, 'o', markersize=14, markeredgecolor='black')
        ax[0].set_xlim([0, system.box_length])
        ax[0].set_ylim([0, system.box_length])
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        self.step_text = ax[0].text(0.98, 0.02, '', transform=ax[0].transAxes,
                                       fontsize=12, horizontalalignment='right', verticalalignment='bottom')
        ax[1].plot([0] * 20)
        ax[1].set_ylabel('Temperature', fontsize=16)
        ax[1].set_xlabel('Step', fontsize=16)
        self.temp_text = ax[1].text(0.98, 0.02, 'Temperature={:f}'.format(np.average(system.temp_array)),
                                       transform=ax[1].transAxes, fontsize=12, horizontalalignment='right',
                                       verticalalignment='bottom')

        plt.tight_layout()
        self.ax = ax
        self.fig = fig

    def update(self, particles, system, text):
        self.step_text.set_text(text)

        x3 = np.array([])
        y3 = np.array([])
        for i in range(0, particles.size):
            x3 = np.append(x3, particles[i].xpos)
            y3 = np.append(y3, particles[i].ypos)

        line2 = self.ax[0].lines[0]
        line2.set_ydata(y3)
        line2.set_xdata(x3)

        line1 = self.ax[1].lines[0]
        line1.set_ydata(system.temp_array)
        line1.set_xdata(np.arange(0, len(system.temp_array)))
        self.ax[1].set_xlim(0, len(system.temp_array))
        self.ax[1].set_ylim(np.amin(system.temp_array)-np.amax(system.temp_array) * 0.05,
                               np.amax(system.temp_array)+np.amax(system.temp_array) * 0.05)
        self.temp_text.set_text('Temp={:.3f}+/-{:.3f}'.format(np.average(system.temp_array), np.std(system.temp_array)))

        self.fig.canvas.draw()


class Interactions(object):
    def __init__(self, system):
        fig, ax = plt.subplots(2, 2, figsize=(9, 9))

        ax[0, 1].plot([0] * 20, color='#34a5daff')
        #ax[0, 1].set_xlim([0, system.box_length / 2])
        ax[0, 1].set_ylabel('Force', fontsize=16)
        ax[0, 1].set_xlabel('Step', fontsize=16)
        self.step_text = ax[0, 1].text(0.98, 0.93, 'Time={:.1f}'.format(system.step), transform=ax[0, 1].transAxes,
                                       fontsize=12, horizontalalignment='right', verticalalignment='bottom')
        ax[1, 1].plot([0] * 20, color='#34a5daff')
        ax[1, 1].set_ylabel('Temperature', fontsize=16)
        ax[1, 1].set_xlabel('Step', fontsize=16)

        ax[0, 0].plot([0] * 20, 'o', markersize=14, markeredgecolor='black', color='#34a5daff')
        ax[0, 0].set_xlim([0, system.box_length])
        ax[0, 0].set_ylim([0, system.box_length])
        ax[0, 0].set_xticks([])
        ax[0, 0].set_yticks([])
        #self.temp_text = ax[1, 1].text(0.98, 0.02, 'Temperature={:f}'.format(np.average(system.temp_array)),
        #                               transform=ax[1, 1].transAxes, fontsize=12, horizontalalignment='right',
        #                               verticalalignment='bottom')

        ax[1, 0].plot([0] * 20, color='#34a5daff')
        ax[1, 0].set_ylabel('Pressure', fontsize=16)
        ax[1, 0].set_xlabel('Step', fontsize=16)
        #self.press_text = ax[1, 0].text(0.98, 0.02, 'Pressure={:f}'.format(np.average(system.press_array)),
        #                               transform=ax[1, 0].transAxes, fontsize=12, horizontalalignment='right',
        #                               verticalalignment='bottom')

        plt.tight_layout()
        self.ax = ax
        self.fig = fig
        self.forces = []

    def update(self, particles, system):
        line = self.ax[0, 1].lines[0]
        #self.forces.append(np.sum(system.force_))
        line.set_xdata(np.arange(0, len(system.force_array)))
        line.set_ydata(system.force_array)
        self.ax[0, 1].set_xlim(0, len(system.force_array))

        self.ax[0, 1].set_ylim(np.amin(system.force_array)-np.amax(system.force_array) * 0.05,
                               np.amax(system.force_array)+np.amax(system.force_array) * 0.05)
        self.step_text.set_text('Time={:.1f}'.format(system.time))

        x3 = np.array([])
        y3 = np.array([])
        for i in range(0, particles.size):
            x3 = np.append(x3, particles[i].xpos)
            y3 = np.append(y3, particles[i].ypos)

        line2 = self.ax[0, 0].lines[0]
        line2.set_ydata(y3)
        line2.set_xdata(x3)

        line1 = self.ax[1, 1].lines[0]
        line1.set_ydata(system.temp_array)
        line1.set_xdata(np.arange(0, len(system.temp_array)))
        self.ax[1, 1].set_xlim(0, len(system.temp_array))
        self.ax[1, 1].set_ylim(np.amin(system.temp_array)-np.amax(system.temp_array) * 0.05,
                               np.amax(system.temp_array)+np.amax(system.temp_array) * 0.05)
        #self.temp_text.set_text('Temp={:.3f}+/-{:.3f}'.format(np.average(system.temp_array), np.std(system.temp_array)))

        line3 = self.ax[1, 0].lines[0]
        line3.set_ydata(system.press_array)
        line3.set_xdata(np.arange(0, len(system.press_array)))
        self.ax[1, 0].set_xlim(0, len(system.press_array))
        self.ax[1, 0].set_ylim(np.amin(system.press_array) - np.amax(system.press_array) * 0.05,
                               np.amax(system.press_array) + np.amax(system.press_array) * 0.05)
        #self.press_text.set_text('Pressure={:.0f}+/-{:.0f}'.format(np.average(system.press_array[-100:]),
        #                                                        np.std(system.press_array[-100:])))

        self.fig.canvas.draw()

    def average_rdf(self):
        gr = np.average(self.avgr, axis=0)
        x = self.xgr
        line = self.ax[0, 1].lines[0]
        line.set_xdata(x)
        line.set_ydata(gr)
        self.ax[0, 1].set_ylim([0, np.amax(gr) + 0.5])
        self.step_text.set_text('Average')
