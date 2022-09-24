# Schrodinger's 1D equatiion
# Quantum Mechanical Simulation using Finite-Difference Time-Domain(FDTD) Method
# This script simulates a probability wave in the presence of multiple
# potentials. The simulation is carried out by using the FDTD algorithm applied to the
# Schrodinger's equation.
# 1) Constant potential: Demonstrates a free particle with dispersion.
# 2) Step potential: Demonstrate transmission and reflection.
# 3) Potential barrier: Demonstrates tunneling.

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import pylab
# set pylab to interactive mode so plots update when run outside ipython
plt.ion()
# Utility functions defines a quick Gaussian pulse function to act as an
# envelope to the wave function .

def Gaussian(x,t,sigma):
    """ A Gaussian curve.
        x=Variable
        t=Time shift
        sigma=standard deviation"""
    return np.exp(-(x-t)**2/(2*sigma**2))
def free(npts):
    "Free particle"
    return np.zeros(npts)
def step(npts,v0):
    "Potential step"
    v=free(npts)
    v[npts:]= v0
    return v
def barrier(npts,v0,thickness):
    "Barrier potential"
    v=free(npts)
    v[npts:+npts+thickness]=v0
    return v
def fillax(x,y,*args,**kw):
    """Fill the space between an array of y values and  the x axis.
        All args/kwargs are passed to the pylab.fill function.
        Returns the value of the pylab.fill() call."""
    xx= np.concatenate((x,np.array([x[-1],x[0]],x.dtype)))
    yy= np.concatenate((y,np.zeros(2,y.dtype)))
    return plt.fill(xx,yy,*args,**kw)

# Simulation constants. Be sure to include decimal points on appropriate
# variables so they become floats instead of integers.

N=1200 # Number of spatial points.
T= 5*N # Number of time step.5*N is a nice value for terminating before anything reaches the boundaries.
Tp= 50 # Number of time steps to increment before updating the plot.
dx=1.0e0 # Spatial resolution
m= 1.0e0 # Particle mass
hbar=1.0e0 # Plank's constant
X= dx*np.linspace(0,N,N) # spatial axis
# Potential parameters
v0=1.0e-2 # Potential Amplitude ( Used for steps and barriers)
THCK= 15 # Thickness of the potential barrier
#POTENTIAL ='free'
POTENTIAL ='step'
#POTENTIAL ='barrier'
sigma=40.
x0=round(N/2)-5*sigma # Time-shift
k0=np.pi/20
E=(hbar**2/2.0/m)*(k0**2+0.5/sigma**2)
if POTENTIAL=='free':
    v=free(N)
elif POTENTIAL=='step':
    v=step(N,v0)
elif POTENTIAL=='barrier':
    v=barrier(N,v0,THCK)
else:
    raise ValueError("Unrecognized potential type: %s" % POTENTIAL)
vmax=v.max() # Maximum potential of the domain.
dt= hbar/(2*hbar**2/(m*dx**2)+vmax)   # Critical time
c1=hbar*dt/(m*dx**2)   # constant coefficient1
c2=2*dt/hbar           # constant coefficient2
c2v=c2*v               # pre-compute outside of update loop

print ('One-dimensional Schrodinger equation - Time evolution')
print ('Wavepacket energy: ',E)
print ('Potential type: ',POTENTIAL)
print ('Potential height v0: ',v0)
print ('Barrier thickness: ',THCK)
# Wave functions .Three states represents past,present and future.
psi_r=np.zeros((3,N)) # Real
psi_i=np.zeros((3,N)) # Imaginary
psi_p=np.zeros(N,)    # Observable probability(magnitude-squared of the complex wave function).
# Temporal indexing constant ,used for accessing rows of the wavefunctions.
PA=0 #past
PR=1 #Present
FU=2 #Future
# Initialize wavefunction.A Present-only will "split" with half the
# wavefunction propagating to the left and the other half to the right.
# Including a "past" state will cause it to propagate one way.
xn=range(1,N)
x=X[xn]/dx    # Normalized position coordinate
gg=Gaussian(x,x0,sigma)
cx=np.cos(k0*x)
sx=np.sin(k0*x)
psi_r[PR,xn]=cx*gg
psi_i[PR,xn]=sx*gg
psi_r[PA,xn]=cx*gg
psi_i[PA,xn]=sx*gg
#Initial NormalizaTION OF wavefunctions
# compute the observable probability.
psi_p=psi_r[PR]**2+psi_i[PR]**2
# Normalize the wavefunction so that the total probability in the simulation is equal to 1.
P=dx*psi_p.sum()  # Total probability
nrm=np.sqrt(P)
psi_r /=nrm
psi_i /=nrm
psi_p /=P
# Initialize the figure and axes.
plt.figure()
xmin=X.min()
xmax=X.max()
ymax=1.5*(psi_r[PR]).max()
plt.axis([xmin,xmax,-ymax,ymax])
lineR, =plt.plot(X,psi_r[PR],'b',alpha=0.7,label='Real')
lineI, =plt.plot(X,psi_i[PR],'r',alpha=0.7,label='Imag')
lineP, =plt.plot(X,6*psi_p,'k',label='Probability')
plt.title('Potential height: %.2e' % v0)
# For non-zero Potentials , plot them and shade the classically forbidden region
# in light red,as well as drawing a green line at the wavepacket's total energy
# in the same units the potential is being plotted.
if vmax!=0:
    # Scaling factor for energies , so they fit in the same plot as the wavefunctions
    Efac=ymax/2.0/vmax
    v_plot=v*Efac
    plt.plot(X,v_plot,':k',zorder=0) # Potential line
    fillax(X,v_plot,facecolor='y',alpha=0.2,zorder=0)
    # plot the wavefunction energy,in the same scale as the potential
    plt.axhline(E*Efac,color='g',label='Energy',zorder=1)
plt.legend(loc='lower right')
plt.draw()
# Plotting the E line . Fix it back manually.
plt.xlim(xmin,xmax)
# Direct index assignment is MUCH faster than using a spatial FOR loop,so
# these constant are used in the update equations. Remember that python uses zero-based indexing
IDX1=range(1,N-1)         # psi[k]
IDX2=range(2,N)           # psi[k+1]
IDX3=range(0,N-2)         # psi[k-1]
for t in range(T+1):
    # Precompute a couple of indexing constants , this speeds up the computation
    psi_rPR=psi_r[PR]
    psi_iPR=psi_i[PR]
    # Apply the update equations
    psi_i[FU,IDX1]=psi_i[PA,IDX1]+ \
                    c1*(psi_rPR[IDX2]-2*psi_rPR[IDX1]+
                        psi_rPR[IDX3])
    psi_i[FU] -= c2v*psi_r[PR]
    psi_r[FU,IDX1]=psi_r[PA,IDX1]- \
                    c1*(psi_iPR[IDX2]-2*psi_iPR[IDX1]+
                        psi_iPR[IDX3])
    psi_r[FU]+=c2v*psi_i[PR]
    # Increment the time steps. PR->PA and FU->PR
    psi_r[PA]=psi_rPR
    psi_r[PR]=psi_r[FU]
    psi_i[PA]=psi_iPR
    psi_i[PR]=psi_i[FU]
    # Only plot after a few iterations to make the simulation run faster.
    if t % Tp == 0:
        # Compute Observable Probability for the plot .
        psi_p=psi_r[PR]**2+psi_i[PR]**2
        # Updates the plots.
        lineR.set_ydata(psi_r[PR])
        lineI.set_ydata(psi_i[PR])
        # Note : That we plot the probability density amplified by a factor so it's a bit easier to see
        lineP.set_ydata(6*psi_p)
        plt.draw()
# So the windows don't auto-close at the end if run outside the ipython
#animation=FuncAnimation(fig,func=psi_p,frames=np.arange(0,10,0.01),interval=10)
plt.ioff()
plt.show()





























        







































    


    























































































    
    
    

