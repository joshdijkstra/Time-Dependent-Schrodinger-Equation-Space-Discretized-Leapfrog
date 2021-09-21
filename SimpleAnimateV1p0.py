# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11  2020

@author: shugh
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# niceFigure example with good reasonable size fonts and TeX fonts
def niceFigure(useLatex=True):
    from matplotlib import rcParams
    plt.rcParams.update({'font.size': 19})
    if useLatex is True:
        plt.rc('text', usetex=True)
        plt.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'
    rcParams['xtick.major.width'] = 1
    rcParams['ytick.major.width'] = 1
    return 

# for simple graphics/animation   
def go():
    cycle=50; ic=0;h=1.e-2
    niceFigure()
    fig = plt.figure(figsize=(10,5))
#    plt.ion()
    ax = fig.add_axes([.18, .18, .33, .66])
    ax.set_ylim(-1.2,1.2)
    ax.set_xlim(-1.2,1.2)
    ax.set_xlabel('$x$ (arb. units)')     # add labels
    ax.set_ylabel('$y$ (arb. units)')
    x, v = 1.0, 0.0         # initial values
    line, = ax.plot( x, v,'x', markersize=12) # Fetch the line object
    T0 = 2.*np.pi
    t=0.
    tpause = 1.e-2 # delay for animation in loop
    plt.pause(2) # pause for 2 seconds before start

    while t<T0*10:        # loop for 10 periods
        x = np.cos(t)
        v = -np.sin(t)
        # so your leapfrog call could be here for example ...
        #xl, vl = leapfrog(oscillator, xl, vl, t, h) # solve this baby
              
        if (ic % cycle == 0): # very simple animate (update data with pause)
            ax.set_title("frame time {}".format(ic))
            line.set_xdata(x)
            line.set_ydata(v)

            plt.draw() # may not be needed
            plt.pause(tpause) # pause to see animation as code v. fast

        t = t + h
        ic=ic+1

go()
