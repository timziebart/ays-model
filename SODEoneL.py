
from __future__ import print_function, division

from scipy.integrate import odeint
import scipy
#from numpy import linspace, shape, mgrid, sqrt
import numpy as np

#from matplotlib.pyplot import plot, show, figure, subplot
import matplotlib.pyplot as plt

#odeint = scipy.integrate.odeint
array = np.array

import sys
import scipy.integrate as integrate

class SODEone:
    """System of first order Ordinary Differential Equationds, simple version

usage:
SODEone(N, RHS)

N: nr of variables
RHS: right-hand side of equation (LHS is the derivative)
    """
    def __init__(self, RHS):
       self.RHS = RHS
       self.initialCond = []
       self.TimeScaling = 1.
       self.trajectory = []

    def eval(self, *Xs):
        return self.RHS(Xs, 0)

    def setInitialCond(self, x0s):
        x0s = np.array(x0s)
        assert x0s.ndim == 1
        self.initialCond = np.array(x0s)

    def isSetInitialCond(self):
        return self.initialCond.any()

    def run(self, T0, T1, stepSize=None, steps = None):
        assert self.isSetInitialCond()
        assert steps == None or stepSize == None, "either you give the number of steps or the step size ... come on ..."

        if stepSize:
            steps = int(T1-T0)/self.TimeScaling/stepSize
        elif not steps:
            steps = 1e5
        
        t = np.linspace(T0/self.TimeScaling, T1/self.TimeScaling, steps)
        
        self.trajectory = odeint(self.RHS, self.initialCond, t)
        self.timeline = t*self.TimeScaling

        return (self.timeline, self.trajectory)

    def plotTrajectory(self, rescale = [], styles = [], labels = [], legends = [], saveToFile=None):
        
        N = np.shape(self.trajectory)[1]

        #print np.shape(self.trajectory)
        if rescale:
            assert N == len(rescale)
        else:
            rescale = np.ones((N, ))

        if labels:
            assert len(labels) == 2 # x and y axis label

        if styles:
            assert len(styles) == N
        else:
            styles = ['b'] * N

        if legends:
            assert len(legends) == N

        #self.trajPlot = plt.figure()
        
        plotList = []
        for i in range(len(rescale)):
            plotList.append( plt.plot(self.timeline, self.trajectory[:, i]*rescale[i], styles[i]) )

        #if legends:
        #    plt.legend( plotList, legends )

        if labels:
            plt.xlabel(labels[0])
            plt.ylabel(labels[1])


        if saveToFile:
            plt.savefig(saveToFile)
            if verbosity:
                print("trajectory plot saved to " + saveToFile)
        #return self.trajPlot



    def plotTrajectoryPS(self, var1, var2, rescale = [], style = None, labels = [], start = 0, stop = None):
        
        N = 2 # only 2 dimensional2 # only 2 dimensional

        if rescale:
            assert N == len(rescale)
        else:
            rescale = np.ones((N, ))

        if labels:
            assert len(labels) == 2 # x and y axis label

        if not style:
            style = 'b'

        if stop:
                plt.plot(self.trajectory[start:stop, var1]*rescale[var1], self.trajectory[start:stop, var2]*rescale[var2], style)
        else:
                plt.plot(self.trajectory[start:, var1]*rescale[var1], self.trajectory[start:, var2]*rescale[var2], style)

        if labels:
            plt.xlabel(labels[0])
            plt.ylabel(labels[1])





if __name__ == "__main__":
    
    def testsin(x,t):
        #return x # gives exp
        return array([-x[1], x[0]]) # gives sin and cos

    odes = SODEone(testsin)
    odes.setInitialCond(1,0)
    odes.run(0,10)
    odes.plotTrajectory(styles = ["r--", "g"])
    
    plt.show()

