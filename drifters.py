import numpy as np
from pylab import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# This is a simple skeleton code just to help beginners get going
# It creates collisionles particles moving at random in a cubic box with edges at -1 and +1

# For reproducibility, set a seed for randomly generated inputs. Change to your favourite integer.
np.random.seed(4080)

# Choose projection for the main panel
project_3d = False

# Set the number of spatial dimensions (at least 2)
Nd = 3

# Set the number of particles to simulate
Np = 100

# Set the total number of timesteps
Nt = 100

# Set how long the animation should dispay each timestep (in milliseconds).
frame_duration = 100

# Set the maximum drift velocity, in units of position units per timestep
v_max= 0.05

#Set softening length, in natural units of length.
slen = 0.005

#Set mass of all particles:
mass = 0.00001

#Set number of correlation function bins:
ncorrbins = 20

#Set initial positions at random within box:
position = 1-2*np.random.random((Nd,Np))

#Set random positions for correlation function:
rand_position = 1-2*np.random.random((Nd,Np))

# Set initial velocities to be random fractions of the maximum
velocity = v_max*(1-2*np.random.random((Nd,Np)))

def acceleration(p):
    a = np.random.random((Nd,Np))
    for i in range(len(p[0])):
        for k in range(len(a)):
            a[k][i] = 0.0
        for j in range(len(p[0])):
            r = 0
            if i != j:
                for k in range(len(a)):
                    r += (p[k][i] - p[k][j])**2
                for k in range(len(a)):
                    a[k][i] += -(1 - (slen/np.sqrt(r))**3)*mass*(p[k][i] - p[k][j])/(np.sqrt(r)**3)
    return a


# Create a function to apply boundary conditions
def apply_boundary(p):
    for i in range(len(p)):
        for j in range(len(p[0])):
            if abs(p[i][j]) > 1:
                p[i][j] = -p[i][j]
    return p

#Finds the separation between 2 masses.
def separation(p1,p2):
    sep = []
    for i in range(len(p1[0])):
        for j in range(len(p2[0])):
            r = 0
            if i != j:
                for k in range(len(p1)):
                    r += (p1[k][i] - p2[k][j])**2
                sep.append(np.sqrt(r))
    return sep

#Correlation Function
def correlation_func(d,r,x,dx):
    #These vectors store the separations between data, random points and data
    #and random points.
    DDsep = separation(d,d)
    DRsep = separation(d,r)
    RRsep = separation(r,r)

    #These vectors have the number of masses binned at a given separations.
    DD = np.zeros(len(x))
    DR = np.zeros(len(x))
    RR = np.zeros(len(x))

    for i in range(len(DDsep)):
        for j in range(len(x)-1):
            #If there is a separation within a given bin, add 1 to the count.
            if DDsep[i] > xb[j] and DDsep[i] < xb[j+1]:
                DD[j] += 1
            if DRsep[i] > xb[j] and DRsep[i] < xb[j+1]:
                DR[j] += 1
            if RRsep[i] > xb[j] and RRsep[i] < xb[j+1]:
                RR[j] += 1

    corr = np.divide(np.add(DD,np.add(-2*DR,RR)),RR)
    for i in range(len(corr)):
        #Setting any nans or infs to 0 (for future calculations).
        if np.isnan(corr[i]) or np.isinf(corr[i]):
            corr[i] = 0
    return corr

def power_spec(xb,kb,cor):
    pk = []
    #Integrating for a range of k-values (given by kb):
    pk.append([np.trapz(2.0*np.pi*np.multiply(np.multiply(xb,np.sin(i*xb)),cor)*(1.0/i),xb) for i in kb])
    return pk


plt.ion() # Set interactive mode on
fig = figure(figsize=(12,6)) # Create frame and set size
subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95,wspace=0.15,hspace=0.2)
# Create one set of axes as the left hand panel in a 1x2 grid
if project_3d:
    ax1 = subplot(121,projection='3d') # For very basic 3D projection
else:
    ax1 = subplot(121) # For normal 2D projection
xlim(-1,1)  # Set x-axis limits
ylim(-1,1)  # Set y-axis limits

if project_3d:
    points, = ax1.plot([],[],[],'o',markersize=4)  ## For 3D projection
else:
    points, = ax1.plot([],[],'o',markersize=4) ## For 2D projection

ax2 = subplot(222) # Create second set of axes as the top right panel in a 2x2 grid
xmax = 6 # Set xaxis limit
xlabel('k')
ylabel('P(k)')
dk=0.1 # Set width of x-axis bins
kb = np.arange(0,50.0,dk)  # Set x-axis bin edges
kb[0] = 1e-6 # Shift first bin edge by a fraction to avoid showing all the zeros (a cheat, but saves so much time!)
xlim(0,kb[-1])
ylim(-3.0,3.0) # Reasonable guess for suitable yaxis scale
line1, = ax2.plot([],[]) # Define a command that plots a line in this panel

ax4 = plt.subplot(224) # Create last set of axes as the bottom right panel in a 2x2 grid
xlabel('Separation')
ylabel(r'$\xi_{LS}$')
dx=0.05 # Set width of x-axis bins
xb = np.arange(0,1.0+dx,dx)  # Set x-axis bin edges
xb[0] = 1e-6 # Shift first bin edge by a fraction to avoid showing all the zeros (a cheat, but saves so much time!)
ylim(-100,100) # Reasonable guess for suitable yaxis scale
line2, = ax4.plot([],[],drawstyle='steps-post')

# Define procedure to update positions at each timestep
def update(i):
    global position,velocity # Get positions and velocities
    velocity += acceleration(position)
    position += velocity # Increment positions according to their velocites
    position = apply_boundary(position) # Apply boundary conditions
    points.set_data(position[0,:], position[1,:]) # Show 2D projection of first 2 position coordinates
    correlation = correlation_func(position,rand_position,xb,dx)
    power = power_spec(xb,kb,correlation)
    if project_3d:
        points.set_3d_properties(position[2,:])  ## For 3D projection
    line1.set_data(kb,power) # Set the new data for the line in the 2nd panel
    line2.set_data(xb,correlation) # Set the new data for the line in the 2nd panel
    return points,line1,line2 # Plot the points and the line

# Create animation
# https://matplotlib.org/api/_as_gen/matplotlib.animation.FuncAnimation.html
ani = animation.FuncAnimation(fig, update, frames=Nt,interval = frame_duration)

plt.show()
# To save as an mpeg file try:
ani.save("drifters.mp4")
