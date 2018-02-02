import numpy as np
import matplotlib.pyplot as plt
from pylj import md

class liveplot(object):
    def __init__(self, system):
        fig,ax = plt.subplots(2, 2, figsize=(10, 10))

        ax[0, 1].plot([0]*20)
        ax[0, 1].set_xlim([0, system.box_length/2])
        ax[0, 1].set_xticks([])
        ax[0, 1].set_yticks([])
        ax[0, 1].set_ylabel('$g(r)$', fontsize=20)
        ax[0, 1].set_xlabel('$r$', fontsize=20)
        self.step_text = ax[0, 1].text(0.98, 0.95, 'Step={:d}'.format(system.step), transform=ax[0, 1].transAxes,
                                       fontsize=16, horizontalalignment='right', verticalalignment='center')

        ax[1, 1].plot([0]*20)
        ax[1, 1].set_xticks([])
        ax[1, 1].set_yticks([])
        ax[1, 1].set_ylabel('log$(I(q))$', fontsize=20)
        ax[1, 1].set_xlabel('$q$', fontsize=20)

        ax[0, 0].plot([0]*20, 'o', markersize=15, markeredgecolor='black')
        ax[0, 0].set_xlim([0, system.box_length])
        ax[0, 0].set_ylim([0, system.box_length])
        ax[0, 0].set_xticks([])
        ax[0, 0].set_yticks([])

        ax[1, 0].plot([0] * 20)
        #ax[1, 0].set_xticks([])
        #ax[1, 0].set_yticks([])
        ax[1, 0].set_ylabel('Temperature', fontsize=20)
        ax[1, 0].set_xlabel('Step', fontsize=20)

        self.ax = ax
        self.fig = fig

    def update(self, x, y, x2, y2, particles, system):
        line = self.ax[0,1].lines[0]
        line.set_xdata(x)
        line.set_ydata(y)
        self.ax[0, 1].set_ylim([0, np.amax(y)+0.5])
        self.step_text.set_text('Step={:d}'.format(system.step))

        line1 = self.ax[1,1].lines[0]
        line1.set_xdata(x2)
        line1.set_ydata(y2)
        self.ax[1,1].set_ylim([np.amin(y2)-np.amax(y2)*0.05, np.amax(y2)+np.amax(y2)*0.05])
        self.ax[1,1].set_xlim([np.amin(x2), np.amax(x2)])

        x3 = np.array([])
        y3 = np.array([])
        for i in range(0, particles.size):
            x3 = np.append(x3, particles[i].xpos)
            y3 = np.append(y3, particles[i].ypos)

        line2 = self.ax[0, 0].lines[0]
        line2.set_ydata(y3)
        line2.set_xdata(x3)

        line3 = self.ax[1, 0].lines[0]
        line3.set_ydata(system.temp_array)
        line3.set_xdata(np.arange(0, len(system.temp_array)))
        self.ax[1, 0].set_xlim(0, len(system.temp_array))
        self.ax[1, 0].set_ylim(np.amin(system.temp_array)-np.amax(system.temp_array) * 0.05,
                               np.amax(system.temp_array)+np.amax(system.temp_array) * 0.05)

        self.fig.canvas.draw()