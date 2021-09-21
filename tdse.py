# LEIRE LARIZGOITIA ARCOCHA (20190817)
"""
@author: Leire
"""

from scipy.sparse import csr_matrix
from scipy.integrate import quad
import matplotlib.pyplot as plt
import math as m
import numpy as np
import timeit

plt.rcParams.update({'font.size': 17})

def Leapfrog (f , A, R0, I0, x, dt):
    dt2 = dt/2.
    R1 = R0 + dt2* np. asarray(f (0, x, R0 , I0, A))
    I1 = I0 + dt* np. asarray(f (1, x, R1 , I0, A))
    R1 = R1 + dt2* np. asarray(f (0, x, R0 , I1, A))
    return R1 , I1

def derivs (id, x, R, I, A): #FOR LEAPFROG
    if (id == 0): # gen. velocity dR/dt
        tmp = -b*I # 1st/3rd term
        dydt = (2*b + V(x))*I # 2nd term
        dydt [1:-1] += tmp[:-2] + tmp[2:] # add $b\ psi_ {j-1}, b\ psi_ {j+1}$ to grid $j$
        dydt [0] += tmp[-1] + tmp[1] # 1st point , periodic BC
        dydt [-1] += tmp[-2] + tmp[0] # last point , periodic BC
    else : # gen. acceleration dI/dt
        tmp = b*R # 1st/3rd term
        dydt = - (2*b + V(x))*R # 2nd term
        dydt [1:-1] += tmp[:-2] + tmp[2:]
        dydt [0] += tmp[-1] + tmp[1] # 1st point , periodic BC
        dydt [-1] += tmp[-2] + tmp[0] # last point , periodic BC
    return dydt

def derivs_mat (id, x, R, I, A): #FOR LEAPFROG
    if (id == 0): # gen. velocity dR/dt
        dydt = np.dot(A,I)
    else : # gen. acceleration dI/dt
        dydt = -np.dot(A,R)
    return dydt

def V(x): #potential
    return 0.5*x**2

def trap(fa, fb, h): #integral
    II = (fa + fb)*h/2.
    return II

#GLOBAL
n=1000
x0=-5
o=0.5 #sigma
k0=0 #So the intial wavepacket condition is real!
h =20./np.real(n-1)
dt=0.5*h**2 #STABILITY CONDITION
T0 = 2*np.pi
b=1/(2*h**2)

def Q1_a(): # m = w0 = k = 1
    start = timeit.default_timer()  # start timer for solver
    tmax =  2*T0
    tlist = np.arange(0,tmax,dt)

    x=np.empty(n)
    R0=np.empty(n)
    I0=np.empty(n)
    R=np.empty(n)
    I=np.empty(n)
    phi=np.empty([n,n])

    for i in range(0,n): #initial wavepacket
        x[i] = -10. + h*(i)
        R0[i] = (o*np.sqrt(np.pi))**(-0.5) * np.exp(-(x[i]-x0)**2/(2*o**2))
    I0[:]=0.

    for k in range(len(tlist)): #loop over time
        R,I = Leapfrog(derivs,0,R0,I0,x,dt)
        R0,I0 = R,I

        # ANIMATION
        if k/200 == k//200:
            plt.clf()
            plt.plot(x, (np.sqrt(2.)*x/10)**2/2.)
            plt.plot(x, (R0**2 + I0**2)**2)
            plt.ylim(0,1.5)
            plt.xlabel('$Position\ (m)$')
            plt.ylabel('$|\u03A8 (x,t)|^2 (a.u.)$')
            plt.title('{:.1f} ms'.format(k*dt))
            plt.pause(.1e-20)

        # PLOT SAVING
        if k == 3 or k==20000 or k== 40000 or k==62000:
            plt.plot(x, (np.sqrt(2.)*x/10)**2/2.)
            plt.plot(x, (R0**2 + I0**2)**2)
            plt.ylim(0,1.5)
            plt.xlabel('$Position\ (m)$')
            plt.ylabel('$|\u03A8 (x,t)|^2 (a.u.)$')
            plt.title('{:.1f} s'.format(k*dt))
            #plt.savefig('./Q1_a.pdf', format='pdf', dpi=1200,bbox_inches = 'tight')
            plt.show()
        """
        cs = plt.contourf(tlist, x,(R0**2 + I0**2)**2, levels=[10, 30, 50],colors=['#808080', '#A0A0A0', '#C0C0C0'], extend='both')
        cs.cmap.set_over('yellow')
        cs.cmap.set_under('blue')
        cs.changed()
        """

    stop = timeit.default_timer()
    print ("Silicing approach time", stop - start)


def Q1_b(): # m = w0 = k = 1
    start = timeit.default_timer()  # start timer for solver

    tmax =  2*T0
    tlist = np.arange(0,tmax,dt)

    aa=np.empty([n,n])

    x=np.empty(n)
    R0=np.empty(n)
    I0=np.empty(n)
    R=np.empty(n)
    I=np.empty(n)


    for i in range(0,n): #initial wavepacket
        x[i] = -10. + h*(i)
        R0[i] = (o*np.sqrt(np.pi))**(-0.5) * np.exp(-(x[i]-x0)**2/(2*o**2))
    I0[:]=0.


    for i in range(1,n):
        for j in range(1,n):
            if (j==i):
                aa[i,j]=1/h**2 + V(x[i])

            elif ((j==i-1) or (j==i+1)):
                aa[i,j]= -1/(2*h**2)
            else:
                aa[i,j]=0.

    aa_sparse = csr_matrix(aa, (n, n)).toarray() #sparse the aa matrix

    for k in range(len(tlist)): #loop over time
        #R,I = Leapfrog(derivs_mat,aa_sparse,R0,I0,x,dt)  #sparse matrix solver
        R,I = Leapfrog(derivs_mat,aa,R0,I0,x,dt) #full matrix solver
        R0,I0 = R,I

    stop = timeit.default_timer()
    #print ("Sparse matrix solver time", stop - start)
    print ("Full matrix solver time", stop - start)

def Q1_c(): # m = w0 = k = 1 = hbar
    tmax =  2*T0
    tlist = np.arange(0,tmax,dt)

    x=np.empty(n)
    R0=np.empty(n)
    I0=np.empty(n)
    R=np.empty(n)
    I=np.empty(n)

    V_x=[]
    T_x=[]
    m_V=np.empty(n-1)
    m_T=np.empty(n)

    for i in range(0,n): #initial wavepacket
        x[i] = -10. + h*(i)
        R0[i] = (o*np.sqrt(np.pi))**(-0.5) * np.exp(-(x[i]-x0)**2/(2*o**2))
    I0[:]=0.

    for k in range(len(tlist)): #loop over time
        R,I = Leapfrog(derivs,0,R0,I0,x,dt)
        R0,I0 = R,I
        for i in range(0,n-1):
            m_V[i]=(R0[i]**2 + I0[i]**2)*V(x[i])
            m_T[i]= R0[i+1]**2 + I0[i+1]**2 + R0[i]**2 + I0[i]**2 -2*R0[i]*R0[i+1] -2*I0[i]*I0[i+1]

        V_x.append(np.trapz(m_V,dx=h))
        T_x.append(np.trapz(m_T**2, dx=h))

    V_exp = (np.trapz(V_x,dx=dt))
    T_exp = (np.trapz(T_x, dx=dt))

#being an harmonic potential n=2,
# so, 2* T_expectation = 2* V_expectation
# So, both exponentials should be equal
    print(V_exp)
    print(T_exp)
    print(T_exp - V_exp)
#SILING APPROACH
#Q1_a()
#MATRIX SOLVER
#Q1_b()

Q1_c()
#END OF CODE
