# Josh Dykstra Assignment 4


########################################################################
# To select which part of the assignment to run, comment out where the
# question function is called e.g #Q2() to stop the second question being
# run. The code can be a bit tepromental with the sizes of the memory so
# the option to select the size of the grid is given to the user.
#
########################################################################



import matplotlib.pyplot as plt
import math as m
import numpy as np
import cmath
import scipy as sc
from scipy.sparse import dia_matrix
import timeit
from matplotlib import ticker, cm

plt.rcParams['axes.grid'] = True

def V(x):
    return 0.5 * x**2

def V2(x):
    # Choose a and b parameters
    a = 1
    b = 1
    return (a*x**4 - b*x**2)


def Leapfrog(f ,R0 , I0  , t , x , h  , potfunc ,A=None):
    # Leapfrog integrator method
    tt = t/2.0
    R1 = R0 + tt* np. asarray(f (0, R0 , I0,x,h, A,potfunc))
    I1 = I0 + t* np. asarray(f (1, R1 , I0,x,h, A,potfunc))
    R1 = R1 + tt* np. asarray(f (0, R0 , I1,x,h, A,potfunc))
    return R1 , I1

def ODE_Sparse(id,R,I, x ,h ,A,potfunc):
    # ODE to be used with Leapfrog for matrix
    if id == 0:
        dydt = np.dot(A,I)
    else:
        dydt = -np.dot(A,R)
    return dydt

def ODE(id,R,I,x,h,A,potfunc):
    # Ode to be used for slicing method
    param_b=1/(2*h**2)
    if (id == 0):
        tmp = -param_b*I
        dydt = (2*param_b + potfunc(x))*I
        dydt [1:-1] += tmp[:-2] + tmp[2:]
        dydt [0] += tmp[-1] + tmp[1]
        dydt [-1] += tmp[-2] + tmp[0]
    else :
        tmp = param_b*R
        dydt = - (2*param_b + potfunc(x))*R
        dydt [1:-1] += tmp[:-2] + tmp[2:]
        dydt [0] += tmp[-1] + tmp[1]
        dydt [-1] += tmp[-2] + tmp[0]
    return dydt

def createA(n,xs,h):
    # Creates the sparse matric for A
    bs = -1/(2*h**2)*np.ones(n)
    aa = np.ones(n)
    for x in range(len(aa)):
        aa[x] = 1/(h**2) + V(xs[x])
    B = dia_matrix((bs,[-1]),shape=(n,n)).toarray() # Diagnals
    Bb = dia_matrix((bs,[1]),shape=(n,n)).toarray()
    C = dia_matrix((aa,[0]),shape=(n,n)).toarray()
    A = B + C + Bb
    return A


def reset(nnn,tlist,xs,x0,sigma,k0):
    # Creates the arrays for the initial conditions
    Rphis = np.zeros(nnn)
    Iphis = np.zeros(nnn)
    for i in range(nnn):
        Rphis[i] = (sigma*m.sqrt(m.pi))**(-0.5) * m.exp((-(xs[i]-x0)**2)/(2*sigma**2))
    Rs  = np.zeros((len(tlist),nnn))
    Is  = np.zeros((len(tlist),nnn))
    Rs[0] , Is[0] = Rphis , Iphis
    return Rphis , Iphis ,Rs , Is

def niceFigure(useLatex=True):
    # Latex Figures
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

def animate(framerate,tlist,n,xs,R,I,potfunc,xlimit,ylimit,scaled = True):
    # Animation function
    cycle = framerate
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_axes([.18, .18, .33, .66])
    line, = ax.plot( xs, (R[0]**2 + I[0]**2)**2)
    ax.set(xlim=(-xlimit,xlimit),ylim=(-ylimit,ylimit))
    if (scaled):
        sf = potfunc(xs[0])
    else:
        sf =1
    potential = ax.plot(xs,potfunc(xs)/sf,label="$V(x)$")
    ax.set_xlabel("$x (a.u.)$")
    ax.set_ylabel("$|\Psi(x,t)|^2$")
    ax.legend(loc="upper right")
    tpause = 1.e-2
    for x in range(len(tlist)):
        if (x % cycle == 0):
            prob= (R[x]**2 + I[x]**2)**2
            #ax.set_title("$frame time {}$".format(x))
            line.set_xdata(xs)
            line.set_ydata(prob)
            plt.pause(tpause)
    plt.clf()

def snapshots(xs,tlist,R,I,potfunc,frameRate,pdframes,title,xlimit,ylimit,scaled=True):
    # Creates a 2x2 grid of snapshots from the animation at chosen timeframes
    x_label = "$x (a.u.)$"
    y_label = "$|\Psi(x,t)|^2$"
    tlen = len(tlist)
    frames = frameRate
    pdframe = pdframes
    pds =[]
    if (scaled):
        sf = potfunc(xs[0])
    else:
        sf =1
    for j in range(len(pdframe)):
        pds.append((R[pdframe[j]*frames]**2 + I[pdframe[j]*frames]**2)**2)
    fig, axs = plt.subplots(2,2)
    axs[0,0].plot(xs,pds[0])
    axs[0,0].plot(xs,potfunc(xs)/sf,label="$V(x)$")
    axs[0,0].legend(loc="lower left", prop={'size': 15})
    axs[0,0].set(xlim=(-xlimit,xlimit),ylim=(-ylimit,ylimit),ylabel=y_label,title="$Frame:"+ str(pdframes[0])+"$")

    axs[0,1].plot(xs,pds[1])
    axs[0,1].plot(xs,potfunc(xs)/sf)
    axs[0,1].set(xlim=(-xlimit,xlimit),ylim=(-ylimit,ylimit),ylabel=y_label,title="$Frame:"+ str(pdframes[1])+"$")

    axs[1,0].plot(xs,pds[2])
    axs[1,0].plot(xs,potfunc(xs)/sf)
    axs[1,0].set(xlim=(-xlimit,xlimit),ylim=(-ylimit,ylimit),xlabel=x_label,ylabel=y_label,title="$Frame:"+ str(pdframes[2])+"$")

    axs[1,1].plot(xs,pds[3])
    axs[1,1].plot(xs,potfunc(xs)/sf)
    axs[1,1].set(xlim=(-xlimit,xlimit),ylim=(-ylimit,ylimit),xlabel=x_label,ylabel=y_label,title="$Frame:"+ str(pdframes[3])+"$")

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.6, hspace=0.50)
    plt.savefig('./Graphs/'+str(title)+'.pdf', format='pdf', dpi=1200,bbox_inches = 'tight')
    plt.show()



def contourPlot(xs,R,I,ts,title,lim,expt=None,pltexpt=True):
    # Creates a contour plot of the probability distribution
    z = np.zeros((len(xs),len(ts)))
    x , y = np.meshgrid(ts,xs)
    for j in range(len(xs)):
        for i in range(len(ts)):
            z[j,i] = (R[i,j]**2 + I[i,j]**2)**2
    levs = [0,0.00625,0.0125,0.025,0.05,0.1,0.2,0.4,0.8,1.6]
    plt.contourf(ts,xs,z,levels=levs)
    if (pltexpt): # Plots the expectation value of <x> on same graph
        plt.plot(ts,expt,color="r",label="$<x>$")
        plt.legend(loc="upper left", prop={'size': 15})
    plt.xlabel("$t (T_0)$")
    plt.ylabel("$x (a.u.)$")
    plt.ylim(-lim,lim)
    plt.title("$Probability Distribution$")

    plt.colorbar()
    plt.savefig('./Graphs/'+str(title)+'.pdf', format='pdf', dpi=1200,bbox_inches = 'tight')
    plt.show()


def Texpect(tlist,xs,R,I):
    # Calculates the Kinetic energy expectation value
    print("Calculating Kinetic Energy Expectation Value")
    dRdx = dIdx = dydx = np.zeros((len(tlist),len(xs)-1))
    integral = np.zeros(len(tlist))
    for j in range(len(tlist)):
        dRdx[j,:] = np.diff(R[j,:],n=1)/np.diff(xs,n=1)
        dIdx[j,:] = np.diff(I[j,:],n=1)/np.diff(xs,n=1)
        dydx[j,:] = (dRdx[j,:]**2 + dIdx[j,:]**2)
        integral[j] = np.trapz(dydx[j,:],xs[1:])
    return 0.5 * integral

def Vexpect(tlist,xs,R,I,potfunc):
    # Calculates the Potential expecation value
    print("Calculating Potential Energy Expectation Value")
    d = np.zeros((len(xs),len(tlist)))
    inte = np.zeros(len(tlist))
    for p in range(len(tlist)):
        psi = R[p,:]**2 +  I[p,:]**2
        d[:,p] = psi * potfunc(xs)
        inte[p] = np.trapz(d[:,p],xs)
    return inte

def Xexpect(tlist,xs,R,I):
     # Calculates the positional expectation value
    print("Calculating Positional Expectation Value")
    #d = np.zeros((len(xs),len(tlist)))
    inte = np.zeros(len(tlist))
    for p in range(len(tlist)):
        psi = R[p,:]**2 +  I[p,:]**2
        #d[:,p] = psi * xs
        R[p,:] = psi * xs # Reusing R to save memory
        inte[p] = np.trapz(R[p,:],xs)
    return inte

def Viral(tlist,xs,R,I,potfunc,dt,h):
    # Calculates Virials Theorem and plots a graph of the differences
    v_exp = Vexpect(tlist,xs,R,I,potfunc)
    t_exp = Texpect(tlist,xs,R,I)
    diff = v_exp - t_exp
    plt.plot(tlist,diff)
    plt.xlabel("$T (T_0)$")
    plt.ylabel("$<T> - <V> (J)$")
    plt.title("$Virial Difference$")
    plt.savefig('./Graphs/Virial.pdf', format='pdf', dpi=1200,bbox_inches = 'tight')
    plt.show()


def Q1a():
    #SDLF
    print("Slicing Method")
    try:
        n = int(input("Choose n: "))
    except ValueError:
        n = 500
        print("Default: " + str(n))
    try:
        T = int(input("Choose number of periods: ")) * 2 * m.pi
    except ValueError:
        T  =  4 * m.pi
        print("Default: 2")
    h = 20/n
    dt = 0.5 * h**2
    xs = np.arange(-10,10,h)
    tlist = np.arange(0,T,dt)/(2*m.pi)
    x0 = -5
    sigma = 0.5
    k0 =  0
    R0, I0 , Rs , Is = reset(n,tlist,xs,x0,sigma,k0) # Resets R and I to initial conditions
    start = timeit.default_timer()
    for y in range(1,len(tlist)):
        Rs[y] , Is[y]
        R , I = Leapfrog(ODE,R0,I0,dt,xs,h,V) # Calls leapfrog integrator
        R0 , I0 = R , I
        Rs[y] , Is[y] = R , I
    finish = timeit.default_timer()
    print("SDLF Solver Time: "+ str(finish-start))
    contourPlot(xs,Rs,Is,tlist,"SDLFContourPlot",6,pltexpt=False) # If memory error reduce n
    snapshots(xs,tlist,Rs,Is,V,300,[3,28,40,50],"SDLFSnapshots",10,1)
    animate(100,tlist,n,xs,Rs,Is,V,10,1)

def Q1b():
    # Sparse matrix approach
    print("Sparse Matrix")
    try:
        n = int(input("Choose n: "))
    except ValueError:
        n = 1000
        print("Default: " + str(n))
    try:
        T = int(input("Choose number of periods: ")) * 2 * m.pi
    except ValueError:
        T  =  4 * m.pi
        print("Default: 2")
    h = 20/n
    dt = 0.5 * h**2
    xs = np.arange(-10,10,h)
    tlist = np.arange(0,T,dt)/(2*m.pi)
    x0 = -5
    sigma = 0.5
    k0 =  0
    param_b=1/(2*h**2)
    # Sparse Matrix Solution
    A = createA(n,xs,h) # Creates the matrix A
    R0, I0 , Rs , Is = reset(n,tlist,xs,x0,sigma,k0)
    start = timeit.default_timer()
    for y in range(1,len(tlist)):
        R, I = Leapfrog(ODE_Sparse,R0,I0,dt,xs,h,V,A)
        R0 , I0 = R , I
        Rs[y] , Is[y] = R , I
    finish = timeit.default_timer()
    print("Sparse Matrix Solver Time (n = "+str(n)+") : "+ str(finish-start))
    #snapshots(xs,tlist,Rs,Is,V,320,[3,28,48,83],"SparseMatrix1000n",10,2)
    #animate(100,tlist,n,xs,Rs,Is,V,10,1)


def Q1c():
    # Virial
    print("Virial")
    try:
        n = int(input("Choose n: "))
    except ValueError:
        n = 750
        print("Default: " + str(n))
    try:
        T = int(input("Choose number of periods: ")) * 2 * m.pi
    except ValueError:
        T  =  4 * m.pi
        print("Default: 2")
    h = 20/n
    dt = 0.5 * h**2
    xs = np.arange(-10,10,h)
    tlist = np.arange(0,T,dt)/(2*m.pi)
    x0 = -5
    sigma = 0.5
    k0 =  0
    R0, I0 , Rs , Is = reset(n,tlist,xs,x0,sigma,k0) # Resets R and I to initial conditions
    for y in range(1,len(tlist)):
        R , I = Leapfrog(ODE,R0,I0,dt,xs,h,V)
        R0 , I0 = R , I
        Rs[y] , Is[y] = R , I
    Viral(tlist,xs,Rs,Is,V,dt,h)


niceFigure()
# Comment out these to not run question 1
Q1a()
Q1b()
Q1c()

def Q2():
    print("Question 2")
    try:
        n = int(input("Choose n: "))
    except ValueError:
        n = 1000
        print("Default: " + str(n))
    try:
        T = int(input("Choose number of periods: ")) * 2 * m.pi
    except ValueError:
        T =  4 * 2 * m.pi
        print("Default: 4")
    lim = 10
    h = lim/n
    dt = 0.5 * h**2

    # Initial conditions
    x0 = -m.sqrt(2)
    sigma = 0.5
    k0 = 0
    tlist  = np.arange(0,T,dt)/(2*m.pi)
    xs = np.arange(-lim/2,lim/2,h)
    R0, I0 , Rs , Is = reset(n,tlist,xs,x0,sigma,k0)
    start = timeit.default_timer()
    for y in range(1,len(tlist)):
        R , I = Leapfrog(ODE,R0,I0,dt,xs,h,V2)
        R0 , I0 = R , I
        Rs[y] , Is[y] = R , I
    finish = timeit.default_timer()
    print("Q2 run Time: "+ str(finish-start))
    title = "Q2ContourPlot"
    xexp = Xexpect(tlist,xs,Rs,Is)
    xlimit = ylimit = 3
    # Choose the frames for the
    snapshots(xs,tlist,Rs,Is,V2,80,[3,25,38,53],"Q2Snapshots",xlimit,ylimit,False)
    animate(100,tlist,n,xs,Rs,Is,V2,xlimit,ylimit,False)
    contourPlot(xs,Rs,Is,tlist,title,2,xexp)
Q2()
