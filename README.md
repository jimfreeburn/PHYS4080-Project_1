# PHYS4080-Project_1
This Python program will generate an n-body simulation based on various hardcoded parameters.
Once run, the script will produce a movie as the 'n-body_sim.mp4.'
The user can modify, as required:

- Whether or not it will output a 3-D projection, project_3d.
- The number of spatial dimensions, Nd.
- The number of particles, Np.
- The total number of timesteps, Nt.
- The duration of each frame of the movie (in ms), frame_duration.
- The initial maximum drift velocity, v_max.
- The softening length, slen.
- The mass of each particle, mass.
- And the number of bins that make up the correlation function.

First, the script sets random positions and velocities for each particle.
Then, with each time-step accelerations are calculated based on gravitational attraction and softening.
Periodic boundary conditions are also applied with each update to the positions.
With each timestep, the correlation function and power spectrum are generated.
The correlation function is approximated with the Landy-Szalay estimator.
Using this, we compute a Fourier transform (using the trapezoidal rule for the integral) to generate our power spectrum.
The movie has 3 panels.
 - On the right, we have a projection of the locations of each particle.
 - On the top left, we have the power spectrum.
 - On the top right, we have the correlation function.

Note: The notebook file is the presentation and only works with JupyterLab.
