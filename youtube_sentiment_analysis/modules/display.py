#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
    Display data module
    @alexandru_grigoras
"""
# Libraries
import operator

import mplcursors
import numpy as np
import scipy
import seaborn as sns
from PyQt5.QtWidgets import QSizePolicy
from matplotlib import ticker
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from wordcloud import WordCloud

# Constants
__all__ = ['DisplayData']
__version__ = '1.0'
__author__ = 'Alexandru Grigoras'
__email__ = 'alex_grigoras_10@yahoo.com'
__status__ = 'release'


class DisplayData(FigureCanvas):
    """Analyse the data and determine sentiment and word frequency"""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        """Class constructor"""
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        self.__parent = parent
        self.__width = width
        self.__height = height

        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        self.__fig_flag = False

    def plot_classifiers(self, x_values, y_values, size, color, x_names, y_names,
                         comments, videos, author, comm_time):
        """Plot the results of the classifiers"""

        # center lines
        self.ax.spines['left'].set_color('none')
        self.ax.spines['right'].set_position('center')
        self.ax.spines['bottom'].set_color('none')
        self.ax.spines['top'].set_position('center')
        self.ax.spines['right'].set_color('gray')
        self.ax.spines['top'].set_color('gray')
        self.ax.spines['left'].set_smart_bounds(True)
        self.ax.spines['bottom'].set_smart_bounds(True)
        self.ax.xaxis.set_ticks_position('bottom')
        self.ax.yaxis.set_ticks_position('left')

        # map the size of points to [10, 200]
        mapped_size = []
        for x in size:
            mapped_size.append(self.__map(x, min(size), max(size), 10, 200))

        # scatter points
        sc = self.ax.scatter(x_values, y_values, c=color, s=mapped_size)

        # labels and limits
        self.ax.set_xlabel(x_names)
        self.ax.set_ylabel(y_names)
        self.ax.set_xlim([-1, 1])
        self.ax.set_ylim([0, 10])
        self.ax.xaxis.set_ticks(np.arange(-1, 1, 0.25))
        self.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
        self.ax.yaxis.set_ticks(np.arange(0, 10, 2))
        self.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
        self.fig.suptitle("Rezultatele algoritmilor de clasificare")

        # Colorbar with label
        divider = make_axes_locatable(self.ax)
        cax = divider.append_axes('right', size='3%', pad=0.05)
        cb = self.fig.colorbar(sc, cax=cax, orientation='vertical')
        cb.set_label('Gradul de încredere')

        c2 = mplcursors.cursor(self.ax)

        @c2.connect("add")
        def _(sel):
            sel.annotation.get_bbox_patch().set(fc="white", alpha=0.9)
            sel.annotation.arrow_patch.set(arrowstyle="simple", fc="black", alpha=0.9)
            sel.annotation.set_text("Video: " + videos[sel.target.index] + "\n" +
                                    "Text: " + comments[sel.target.index] + "\n" +
                                    "Likes: " + str(int(size[sel.target.index])) + "\n" +
                                    "Author: " + author[sel.target.index] + "\n" +
                                    "Time: " + comm_time[sel.target.index])
            sel.annotation.draggable(True)

        self.__fig_flag = True

        self.draw()

    def plot_heatmap(self, x_values, y_values, x_names, y_names):
        """Plot the heatmap for classifiers result"""

        # Define numbers of generated data points and bins per axis.
        n_bins = 8

        # Construct 2D histogram from data using the 'plasma' colormap
        h, xedges, xedges, image = self.ax.hist2d(x_values, y_values, bins=n_bins, cmap='jet', range=[[-1, 1], [0, 10]])

        # Plot a colorbar with label.
        cb = self.fig.colorbar(image)
        cb.set_label('Număr de recenzii')

        # Center lines and limits
        self.ax.spines['left'].set_color('none')
        self.ax.spines['right'].set_position('center')
        self.ax.spines['bottom'].set_color('none')
        self.ax.spines['top'].set_position('center')
        self.ax.spines['right'].set_color('gray')
        self.ax.spines['top'].set_color('gray')
        self.ax.spines['left'].set_smart_bounds(True)
        self.ax.spines['bottom'].set_smart_bounds(True)
        self.ax.xaxis.set_ticks(np.arange(-1, 1, 0.25))
        self.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
        self.ax.yaxis.set_ticks(np.arange(0, 10, 2))
        self.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))

        # Title and labels
        self.fig.suptitle('Harta termografică a rezultatelor clasificării')
        self.ax.set_xlabel(x_names)
        self.ax.set_ylabel(y_names)

        self.__fig_flag = True

        self.draw()

    def plot_word_frequency(self, items):
        """Plot the word frequency"""

        # sort the above items
        sorted_tuples = sorted(items, key=operator.itemgetter(1), reverse=True)
        a = [i[0] for i in sorted_tuples[0:20]]
        b = [i[1] for i in sorted_tuples[0:20]]
        x = scipy.arange(len(b))
        y = scipy.array(b)

        color_space = sns.color_palette('viridis', len(x))
        self.ax.bar(x, y, align='center', color=color_space)
        self.ax.set_xticks(x)
        self.ax.set_xticklabels(a, rotation=45)
        self.ax.set_xlabel('Cuvinte')
        self.ax.set_ylabel('Frecvența')
        self.fig.suptitle('Frecvența cuvintelor din recenzii')

        self.__fig_flag = True

        self.draw()

    def plot_wordcloud(self, tokens):
        """Plot the wordcloud"""

        # Generate a word cloud image
        plain_text = "".join([" " + i if not i.startswith("'") else i for i in tokens]).strip()
        wordcloud = WordCloud(background_color="white", contour_color='steelblue').generate(plain_text)
        self.ax.imshow(wordcloud, interpolation='bilinear')
        self.ax.set_xticks([], [])
        self.ax.set_yticks([], [])
        self.ax.set_xlabel("")
        self.ax.set_ylabel("")
        self.fig.suptitle("Wordcloud")

        self.__fig_flag = True

        self.draw()

    def plot_accuracy(self, results, names):
        """Make a boxplot with data from classifiers accuracy"""

        # boxplot algorithm comparison
        self.fig.suptitle('Acuretețea algoritmilor de clasificare')
        bp = self.ax.boxplot(results, notch=False, patch_artist=True)
        self.ax.set_xlabel('Algoritm de clasificare')
        self.ax.set_ylabel('Acuratețea rezultatelor utilizând metoda Cross-Validation')
        self.ax.set_xticklabels(names)

        # change outline color, fill color and linewidth of the boxes
        for box in bp['boxes']:
            # change outline color
            box.set(color='#7570b3', linewidth=2)
            # change fill color
            box.set(facecolor='#1b9e77')

        # change color and linewidth of the whiskers
        for whisker in bp['whiskers']:
            whisker.set(color='#7570b3', linewidth=2)

        # change color and linewidth of the caps
        for cap in bp['caps']:
            cap.set(color='#7570b3', linewidth=2)

        # change color and linewidth of the medians
        for median in bp['medians']:
            median.set(color='#b2df8a', linewidth=2)

        # change the style of fliers and their fill
        for flier in bp['fliers']:
            flier.set(marker='o', color='#e7298a', alpha=0.5)

        self.draw()

    def clear_plot(self):
        """Clear the plot data"""

        if self.__fig_flag is True:
            self.fig.clf()
            self.ax = self.fig.add_subplot(111)

    @staticmethod
    def __map(value, left_min, left_max, right_min, right_max):
        """Maps a value from one interval [left_min, left_max] to another [right_min, right_max]"""

        # Check intervals
        if right_min >= right_max:
            return right_min
        if left_min >= left_max:
            return right_min

        # Figure out how 'wide' each range is
        left_span = left_max - left_min
        right_span = right_max - right_min

        if left_span == 0:
            return 0

        # Convert the left range into a 0-1 range (float)
        value_scaled = float(value - left_min) / float(left_span)

        # Convert the 0-1 range into a value in the right range.
        return right_min + (value_scaled * right_span)

