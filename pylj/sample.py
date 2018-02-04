import numpy as np
import matplotlib.pyplot as plt


class Scattering(object):
    def __init__(self, system):
        fig, ax = plt.subplots(2, 2, figsize=(9, 9))

        ax[0, 1].plot([0]*20)
        ax[0, 1].set_xlim([0, system.box_length/2])
        ax[0, 1].set_xticks([])
        ax[0, 1].set_yticks([])
        ax[0, 1].set_ylabel('$g(r)$', fontsize=16)
        ax[0, 1].set_xlabel('$r$', fontsize=16)
        self.step_text = ax[0, 1].text(0.98, 0.95, 'Time={:.1f}'.format(system.step), transform=ax[0, 1].transAxes,
                                       fontsize=12, horizontalalignment='right', verticalalignment='center')
        ax[1, 1].plot([0]*20)
        ax[1, 1].set_xticks([])
        ax[1, 1].set_yticks([])
        ax[1, 1].set_ylabel('log$(I(q))$', fontsize=16)
        ax[1, 1].set_xlabel('$q$', fontsize=16)

        ax[0, 0].plot([0]*20, 'o', markersize=14, markeredgecolor='black')
        ax[0, 0].set_xlim([0, system.box_length])
        ax[0, 0].set_ylim([0, system.box_length])
        ax[0, 0].set_xticks([])
        ax[0, 0].set_yticks([])

        ax[1, 0].plot([0] * 20)
        ax[1, 0].set_ylabel('Pressure', fontsize=16)
        ax[1, 0].set_xlabel('Step', fontsize=16)
        self.temp_text = ax[1, 0].text(0.98, 0.05, 'Pressure={:f}'.format(np.average(system.press_array)),
                                       transform=ax[1, 0].transAxes, fontsize=12, horizontalalignment='right',
                                       verticalalignment='center')

        plt.tight_layout()

        self.ax = ax
        self.fig = fig

    def update(self, particles, system):
        hist, bin_edges = np.histogram(system.distances, bins=np.arange(0, 12.5, system.bin_width))
        gr = hist / (system.number_of_particles * (system.number_of_particles / system.box_length ** 2) * np.pi *
                     (bin_edges[:-1] + system.bin_width / 2.) * system.bin_width)
        x = bin_edges[:-1] + system.bin_width / 2

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
        self.temp_text.set_text('Pressure={:.0f}±{:.0f}'.format(np.average(system.press_array[-100:]),
                                                                np.std(system.press_array[-100:]) / 100))


        self.fig.canvas.draw()


class Interactions(object):
    def __init__(self, system):
        fig, ax = plt.subplots(2, 2, figsize=(9, 9))

        ax[0, 1].plot([0] * 20)
        ax[0, 1].set_xlim([0, system.box_length / 2])
        ax[0, 1].set_xticks([])
        ax[0, 1].set_yticks([])
        ax[0, 1].set_ylabel('$g(r)$', fontsize=16)
        ax[0, 1].set_xlabel('$r$', fontsize=16)
        self.step_text = ax[0, 1].text(0.98, 0.95, 'Time={:.1f}'.format(system.step), transform=ax[0, 1].transAxes,
                                       fontsize=12, horizontalalignment='right', verticalalignment='center')
        ax[1, 1].plot([0] * 20)
        ax[1, 1].set_ylabel('Temperature', fontsize=16)
        ax[1, 1].set_xlabel('Step', fontsize=16)

        ax[0, 0].plot([0] * 20, 'o', markersize=14, markeredgecolor='black')
        ax[0, 0].set_xlim([0, system.box_length])
        ax[0, 0].set_ylim([0, system.box_length])
        ax[0, 0].set_xticks([])
        ax[0, 0].set_yticks([])
        self.temp_text = ax[1, 1].text(0.98, 0.05, 'Temperature={:f}'.format(np.average(system.temp_array)),
                                       transform=ax[1, 1].transAxes, fontsize=12, horizontalalignment='right',
                                       verticalalignment='center')

        ax[1, 0].plot([0] * 20)
        ax[1, 0].set_ylabel('Pressure', fontsize=16)
        ax[1, 0].set_xlabel('Step', fontsize=16)
        self.press_text = ax[1, 0].text(0.98, 0.05, 'Pressure={:f}'.format(np.average(system.press_array)),
                                       transform=ax[1, 0].transAxes, fontsize=12, horizontalalignment='right',
                                       verticalalignment='center')

        plt.tight_layout()
        self.ax = ax
        self.fig = fig

    def update(self, particles, system):
        hist, bin_edges = np.histogram(system.distances, bins=np.arange(0, 12.5, system.bin_width))
        gr = hist / (system.number_of_particles * (system.number_of_particles / system.box_length ** 2) * np.pi *
                     (bin_edges[:-1] + system.bin_width / 2.) * system.bin_width)
        x = bin_edges[:-1] + system.bin_width / 2

        line = self.ax[0, 1].lines[0]
        line.set_xdata(x)
        line.set_ydata(gr)
        self.ax[0, 1].set_ylim([0, np.amax(gr) + 0.5])
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
        self.temp_text.set_text('Temp={:.3f}±{:.3f}'.format(np.average(system.temp_array), np.std(system.temp_array)))

        line3 = self.ax[1, 0].lines[0]
        line3.set_ydata(system.press_array)
        line3.set_xdata(np.arange(0, len(system.press_array)))
        self.ax[1, 0].set_xlim(0, len(system.press_array))
        self.ax[1, 0].set_ylim(np.amin(system.press_array) - np.amax(system.press_array) * 0.05,
                               np.amax(system.press_array) + np.amax(system.press_array) * 0.05)
        self.press_text.set_text('Pressure={:.0f}±{:.0f}'.format(np.average(system.press_array[-100:]),
                                                                np.std(system.press_array[-100:])))

        self.fig.canvas.draw()
