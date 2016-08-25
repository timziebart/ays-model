
from __future__ import print_function

import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl

import pickle

import warnings as warn


def plotPhaseSpace( evol, boundaries, steps = 2000, xlabel = "", ylabel = "", colorbar = True, style = {}, alpha = None , maskByCond = None, invertAxes = False, ax = plt, lwspeed = False):
	"""\
plotPhaseSpace( evol, (Xmin, Ymin, Xmax, Ymax), steps = 2000, xlabel = "", ylabel = "", colorbar = True, style = {}, alpha = None , zeroByCond = None, transPlot = "x"):

plot the phase space of the function evol,

Input:
	evol = evolution function, i.e. deriv from Vera
	boundaries = [Xmin, Ymin, Xmax, Ymax] = minimal/maximal values until where the phase space should be plotted
	saveToFile = target file for plot (None = not saved)
	colorbar = show color bar on right side
	style = set a custom style for the streamplot
	alpha = opacity of the stremplot lines
	maskByCond = set parts of the grid to zero if it's not defined for the complete plot (needed for Anderies Model)

	NOT UP-TO-DATE!!!!
"""

	# separate the boundaries
	Xmin, Ymin, Xmax, Ymax = boundaries

	# check boundaries sanity
	assert Xmin < Xmax
	assert Ymin < Ymax

	# build the grid
	X = np.linspace(Xmin, Xmax, steps)
	Y = np.linspace(Ymin, Ymax, steps)

	XY = np.array(np.meshgrid(X, Y))

	# if Condition give, set everything to zero that fulfills it
	if maskByCond:
		mask = maskByCond(XY[0], XY[1])
		XY[0] = np.ma.array(XY[0], mask = mask)
		XY[1] = np.ma.array(XY[1], mask = mask)

## 		dummy0 = np.zeros((steps,steps))
## 		XY[0] = np.where(mask, XY[0], dummy0)
## 		XY[1] = np.where(mask, XY[1], dummy0)

	# calculate the changes ... input is numpy array
	dX, dY = evol(XY,0) # that is where deriv from Vera is mapped to

	if invertAxes:
		data = [Y, X, np.transpose(dY), np.transpose(dX)]
	else:
		data = [X, Y, dX, dY]


	# separate linestyle
	linestyle = None
	if type(style) == dict and "linestyle" in style.keys():
		linestyle = style["linestyle"]
		style.pop("linestyle")

	# do the actual plot
	if style == "dx":
		c = ax.streamplot(*data, color=dX, linewidth=5*dX/dX.max(), cmap=plt.cm.autumn)
	elif style:
            speed = np.sqrt(data[2]**2 + data[3]**2)
            if "linewidth" in style and style["linewidth"] and lwspeed:
                style["linewidth"] = style["linewidth"] * speed/np.nanmax(speed)
##             print speed
##             print np.nanmax(speed)
            c = ax.streamplot(*data, **style)
	else:
		# default style formatting
		speed = np.sqrt(dX**2 + dY**2)
		c = ax.streamplot(*data, color=speed, linewidth=5*speed/speed.max(), cmap=plt.cm.autumn)


	# set opacity of the lines
	if alpha:
		c.lines.set_alpha(alpha)

	# set linestyle
	if linestyle:
		c.lines.set_linestyle(linestyle)

	# add labels if given
	if invertAxes:
		temp = xlabel
		xlabel = ylabel
		ylabel = temp
	if xlabel:
		if ax == plt:
			ax.xlabel(xlabel)
		else:
			ax.set_xlabel(xlabel)
	if ylabel:
		if ax == plt:
			ax.ylabel(ylabel)
		else:
			ax.set_ylabel(ylabel)

	# add colorbar
	if colorbar:
		assert not "color" in style.keys(), "you want a colorbar for only one color?"
		ax.colorbar()

	return c


def loadAx(fromFile, verb = 1):
    verb = int(verb)
    if verb:
            print("loading axes from %s ..."%fromFile , end = " ")

    axs = []
    with open(fromFile, "rb") as f:
            for _ in range(20):
                    try:
                            axs.append(pickle.load(f))
                    except EOFError:
                            break

    if verb:
            print("done")

    return axs

def saveAx(toFile, verb = 1, ax = None):
    verb = int(verb)
    if verb:
            print("saving axes to %s ..."%toFile , end = " ")

    with open(toFile, "wb") as f:
            if ax is None:
                    if verb >= 2:
                            print("from default axis ...", end = " ")
                    pickle.dump(plt.gca(), f, -1)
            elif type(ax) == list:
                    if verb >= 2:
                            print("from several specified axes:")
                    for a in ax:
                            if verb >= 2:
                                    print(repr(ax))
                            pickle.dump(a, f, -1)
                    if verb >= 2:
                            print()
            elif ax:
                    if verb >= 2:
                            print("from specified axis %s ..."%repr(ax), end = " ")
                    pickle.dump(ax, f, -1)
            else:
                    assert False, "I do not understand what you want to pickle"

    if verb:
            print("done")

def savePlot(toFile, verb = 1, fig = None):
    verb = int(verb)
    if verb:
            print("saving plot to %s ... "%toFile , end = "")

    if fig:
            if verb:
                    print("from specified figure ", end = "")
                    if verb >= 2:
                            print(repr(fig), end = " ")
                    print("... ", end = "")

            fig.savefig(toFile)
    else:
            if verb:
                    print("from default figure ... ", end = "")
            plt.savefig(toFile)

    if verb:
            print("done")




