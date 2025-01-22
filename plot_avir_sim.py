"""
Script to generate synthetic virial plots from the Collins+ 2012
simulation data set taken from CATS. This script generates figures 3
and 4 of Krumholz, Lada, & Forbrich (2025).
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from avir_sim_tools import project_quantities, \
    contour_analysis, contour_summary

# Simulation data files at 0.1 and 0.6 free-fall times
fname = ['C12_Beta0.2_256_0000.h5',
         'C12_Beta0.2_256_0010.h5',
         'C12_Beta0.2_256_0030.h5',
         'C12_Beta0.2_256_0060.h5']
times = np.array([ 0, 0.1, 0.3, 0.6 ])

# Set contour levels
clev = np.linspace(0.1, 0.7, 25)

# Set minimum number of pixels to keep; 30 corresponds to a radius of
# ~= 3 pixels
min_pix = 30

# Numerical value of G in simulation unit system where mean box
# density = 1, box size = 1, sound speed = 1
G = 5 * 9**2 / 3

# Read data
data = []
for f in fname:
    fptr = h5py.File(f, 'r')
    data.append( {
        'rho' : np.array(fptr['density']),
        'vx' : np.array(fptr['velocity_x']),
        'vy' : np.array(fptr['velocity_y']),
        'vz' : np.array(fptr['velocity_z'])
        } )
    fptr.close()

# Compute projections
for d in data:
    project_quantities(d)

# Do contour analysis on data
for d in data:
    d['contour_data'] = contour_analysis(d, clev, G, min_pix = min_pix)

# Compute summary statistics
for d in data:
    d['contour_summary'] = contour_summary(d['contour_data'],
                                           group_proj=True,
                                           ncmin=3)

# Set fonts for plots
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=12)

# Create plot window
fig = plt.figure(1, figsize=(6,6))
fig.clf()

# Set limits
avir_lim = np.array([-0.7, 0.7])
cden_lim = np.array([0.075, 0.7])
s2r_lim = np.array([1.7, 3.1])

# First column: avir_obs versus Sigma
for i in range(len(times)):

    # Create axis
    ax = fig.add_subplot(len(times),2,2*i+1)

    # Make plots
    ax.plot(clev,
            np.log10(data[i]['contour_summary']['avir_mean']),
            color='C0')
    ax.fill_between(clev,
                    np.log10(data[i]['contour_summary']['avir_lo']),
                    np.log10(data[i]['contour_summary']['avir_hi']),
                    color='C0', alpha=0.3)
    ax.plot(cden_lim, np.log10(1)*np.ones(2), 'k--', lw=1, alpha=0.5)
    ax.plot(cden_lim, np.log10(2)*np.ones(2), 'k--', lw=1, alpha=0.5)

    # Add labels
    if i < 3:
        ax.text(0.47, -0.53, r'$t/t_\mathrm{{ff}} = {:3.1f}$'.format(times[i]),
                bbox={'edgecolor' : 'black', 'facecolor' : 'none'})
    else:
        ax.text(0.47, 0.46, r'$t/t_\mathrm{{ff}} = {:3.1f}$'.format(times[i]),
                bbox={'edgecolor' : 'black', 'facecolor' : 'none'})
    if i == 0:
        ax.text(0.5, np.log10(1), r'$\alpha_\mathrm{vir,obs} = 1$',
                fontdict = {'size' : 8}, va='bottom')
        ax.text(0.5, np.log10(2), r'$\alpha_\mathrm{vir,obs} = 2$',
                fontdict = {'size' : 8}, va='bottom')

    # Adjust labels and axes
    ax.set_xlim(cden_lim)
    ax.set_ylim(avir_lim)
    if i == len(times)-1:
        ax.set_xlabel(r'$\log\Sigma$')
    else:
        ax.set_xticklabels('')
    ax.set_ylabel(r'$\log\alpha_\mathrm{vir,obs}$')

# Second column: sigma^2/R versus Sigma
for i in range(len(times)):

    # Create axis
    ax = fig.add_subplot(len(times),2,2*i+2)

    # Make plots
    ax.plot(clev,
            np.log10(data[i]['contour_summary']['s2r_mean']),
            color='C0')
    ax.fill_between(clev,
                    np.log10(data[i]['contour_summary']['s2r_lo']),
                    np.log10(data[i]['contour_summary']['s2r_hi']),
                    color='C0', alpha=0.3)
    ax.plot(cden_lim,
            np.log10(1*G*10.**cden_lim*np.pi/5), 'k--', lw=1, alpha=0.5)
    ax.plot(cden_lim,
            np.log10(2*G*10.**cden_lim*np.pi/5), 'k--', lw=1, alpha=0.5)

    # Add labels
    if i == 0:
        ax.text(0.5, np.log10(1*G*10.**0.5*np.pi/5),
                r'$\alpha_\mathrm{vir,obs} = 1$',
                rotation=15,
                fontdict = {'size' : 8}, va='bottom')
        ax.text(0.5, np.log10(2*G*10.**0.5*np.pi/5),
                r'$\alpha_\mathrm{vir,obs} = 2$',
                rotation=15,
                fontdict = {'size' : 8}, va='bottom')

    # Adjust labels and axes
    ax.set_xlim(cden_lim)
    ax.set_ylim(s2r_lim)
    if i == len(times)-1:
        ax.set_xlabel(r'$\log\Sigma$')
    else:
        ax.set_xticklabels('')
    ax.set_ylabel(r'$\log\langle\sigma^2\rangle/R_\mathrm{eff}$')

# Adjust spacing
plt.subplots_adjust(top=0.95, right=0.95, hspace=0.05, wspace=0.35)

# Save
plt.savefig('avir_sim.pdf')


# Make a second plot illustrating the procedure
fig = plt.figure(2, figsize=(3.5,6.25))
fig.clf()

# Log column density limits
cdmin = -1
cdmax = 1

# Contour level to illustrate
clevex = 0.25

# Create upper image
ax1 = fig.add_subplot(2,1,1)
im = ax1.imshow(np.log10(data[1]['cden']['z']).T, origin='lower',
                aspect='equal', extent=(0,1,0,1),
                vmin=cdmin, vmax=cdmax,
                cmap='coolwarm')
ax1.contour(np.log10(data[1]['cden']['z']).T, levels=[clevex],
            extent=(0,1,0,1), linewidths=[0.5])
ax1.text(0.65, 0.9, r'$t/t_\mathrm{ff} = 0.1$',
         bbox={'facecolor' : 'white', 'alpha' : 0.5 })
ax1.set_xticklabels([])
ax1.set_ylabel(r'$y$')

# Create lower image
ax2 = fig.add_subplot(2,1,2)
ax2.imshow(np.log10(data[3]['cden']['z']).T, origin='lower',
           aspect='equal', extent=(0,1,0,1),
           vmin=cdmin, vmax=cdmax, cmap='coolwarm')
ax2.contour(np.log10(data[3]['cden']['z']).T, levels=[clevex],
            extent=(0,1,0,1), linewidths=[0.5])
ax2.text(0.65, 0.9, r'$t/t_\mathrm{ff} = 0.6$',
         bbox={'facecolor' : 'white', 'alpha' : 0.5 })
ax2.set_ylabel(r'$y$')
ax2.set_xlabel(r'$x$')

# Adjust spacing
fig.subplots_adjust(left=0.1, top=0.88, right=1, hspace=0.07)

# Add color bar
fig.draw_without_rendering()  # Needed to force coordinates to update
coord1 = fig.transFigure.inverted().transform(
    ax1.transData.transform([0,1.03]))
coord2 = fig.transFigure.inverted().transform(
    ax1.transData.transform([1,1.08]))
axc = fig.add_axes((coord1[0], coord1[1],
                    coord2[0]-coord1[0],
                    coord2[1]-coord1[1]))
fig.colorbar(im, cax=axc, orientation='horizontal')
axc.xaxis.set_ticks_position('top')
axc.xaxis.set_label_position('top')
axc.set_xlabel(r'$\log\Sigma$')

# Save
plt.savefig('sim_contour.pdf')
