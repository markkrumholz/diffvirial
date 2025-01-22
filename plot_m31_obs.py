"""
Script to plot observed virial diagram for M31 GMCs. This script
generates figures 5, 6, and 7 of Krumholz, Lada, & Forbrich (2025).
"""

import astropy.units as u
from astropy.constants import G
from astropy.io import ascii as asc
import numpy as np
import matplotlib.pyplot as plt


# Read data
rawdata = asc.read('m31gmcs.csv')

# Break up data by source
ks = rawdata.keys()[0]
sources = rawdata[ks]
source_list = np.unique(sources)
data = { }
for s in source_list:
    idx = rawdata[ks] == s
    data[str(s)] = {
        'AV' : np.array(rawdata['Av_i (mag)'][idx]),
        'M' : rawdata['Mass_i (Msun)'][idx] * u.Msun,
        'R' : rawdata['R_i (pc)'][idx] * u.pc,
        'sigma12' : rawdata['sigma12_i (km/s)'][idx] * u.km/u.s,
        'sigma13' : (rawdata['sigma13_i'][idx] *
                     (rawdata['13CO_S/N'][idx] > 15)) * u.km/u.s,
        'Sigma' : rawdata['SD (Mo/pc^2)'][idx] * u.Msun/u.pc**2
    }

# Discard sources with < 5 contour levels available
data = { k : data[k] for k in data.keys()
         if len(data[k]['Sigma']) >= 5 }

# Compute derived quantities
for k in data.keys():
    data[k]['logSigma'] = np.log10(data[k]['Sigma'] /
                                   (u.Msun/u.pc**2))
    data[k]['avir'] = (5 * data[k]['sigma12']**2 / 
                       (np.pi * data[k]['R'] * G * data[k]['Sigma']))\
                       .to('')
    data[k]['logs2r'] = np.log10((
        data[k]['sigma12']**2 / data[k]['R']) /
                                 (u.km**2 / u.s**2 / u.pc))
    data[k]['avir13'] = (5 * data[k]['sigma13']**2 / 
                       (np.pi * data[k]['R'] * G * data[k]['Sigma']))\
                       .to('')
    data[k]['logs2r13'] = np.log10((
        data[k]['sigma13']**2 / data[k]['R']) /
                                 (u.km**2 / u.s**2 / u.pc))
sources = list(data.keys())

# Mark sources by whether they have rising high-Sigma tails
for s in sources:
    if data[s]['avir'][-1] > data[s]['avir'][-2] and \
       data[s]['avir'][-2] > data[s]['avir'][-3] and \
       data[s]['avir'][-3] > data[s]['avir'][-4] and \
       data[s]['avir'][-4] > data[s]['avir'][-5] and \
       data[s]['avir'][-1] > np.mean(data[s]['avir']):
        data[s]['rising'] = True
    else:
        data[s]['rising'] = False
    data[s]['avir_mean'] = np.mean(data[s]['avir'])
    avir13 = data[s]['avir13'][data[s]['avir13'] > 0]
    data[s]['avir13_mean'] = np.mean(avir13)
    if len(avir13) < 5:
        data[s]['rising13'] = None
    elif avir13[-1] > avir13[-2] and \
       avir13[-2] > avir13[-3] and \
       avir13[-3] > avir13[-4] and \
       avir13[-4] > avir13[-5] and \
       avir13[-1] > np.mean(avir13):
        data[s]['rising13'] = True
    else:
        data[s]['rising13'] = False
    if data[s]['rising13'] is None:
        data[s]['rising_match'] = None
    else:
        data[s]['rising_match'] = data[s]['rising'] == data[s]['rising13']

# Compute statistics
ntot = len(sources)
mtot = np.sum([data[s]['M'][0].to(u.Msun).value
               for s in sources]) * u.Msun
nrise = np.sum([data[s]['rising'] for s in sources])
mrise = np.sum([data[s]['rising'] * data[s]['M'][0].to(u.Msun).value
                for s in sources]) * u.Msun
nrise13 = np.sum([data[s]['rising13'] == True for s in sources])
ntot13 = np.sum([data[s]['rising13'] != None for s in sources])
mtot13 = np.sum([(data[s]['rising13'] != None) * \
                 data[s]['M'][0].to(u.Msun).value
                 for s in sources]) * u.Msun
mrising13 = np.sum([(data[s]['rising13'] == True) * \
                    data[s]['M'][0].to(u.Msun).value
                    for s in sources]) * u.Msun

# Set fonts for plots
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=12)

# Plot setup
avir_lim = [-0.5, 1.7]
avirrel_lim = [-0.3, 0.4]
avir13_lim = [-0.45, 0.5]
cden_lim = [1.3, 2.8]
s2r_lim = [-1.2, 1.5]
s2r13_lim = [-1.4, 0.4]
ms = 2

# Create plot window
fig = plt.figure(1, figsize=(4,5))
fig.clf()

# Observers virial diagram

# Rising
ax = fig.add_subplot(3,2,1)
for s in sources:
    if data[s]['rising']:
        ax.plot(data[s]['logSigma'], data[s]['logs2r'],
                marker='o', markersize=ms)
avir1 = np.log10(
    (np.pi/5) * G * 10.**np.array(cden_lim) * u.Msun / u.pc**2 /
    (u.km**2 / u.s**2 / u.pc))
ax.plot(cden_lim, avir1, 'k--', alpha=0.3)
ax.plot(cden_lim, avir1 + np.log10(2), 'k--', alpha=0.3)
ax.set_xlim(cden_lim)
ax.set_ylim(s2r_lim)
ax.set_xticklabels('')
ax.set_ylabel(r'$\log\sigma^2/R$ [km$^2$ s$^{-2}$ pc$^{-1}$]')
ax.set_title('Rising (N={:d})'.format(nrise),
             size=12)

# Flat/falling
ax = fig.add_subplot(3,2,2)
for s in sources:
    if not data[s]['rising']:
        ax.plot(data[s]['logSigma'], data[s]['logs2r'],
                marker='o', markersize=ms)
ax.plot(cden_lim, avir1, 'k--', alpha=0.3)
ax.plot(cden_lim, avir1 + np.log10(2), 'k--', alpha=0.3)
ax.set_xlim(cden_lim)
ax.set_ylim(s2r_lim)
ax.set_yticklabels('')
ax.set_xticklabels('')
ax.set_title('Flat/falling (N={:d})'.format(ntot-nrise), size=12)

# alpha_vir,obs versus sigma

# Rising sources
ax = fig.add_subplot(3,2,3)
for s in sources:
    if data[s]['rising']:
        ax.plot(data[s]['logSigma'], np.log10(data[s]['avir']),
                marker='o', markersize=ms)
ax.plot(cden_lim, np.zeros(2), 'k--', alpha=0.3)
ax.plot(cden_lim, np.zeros(2) + np.log10(2), 'k--', alpha=0.3)
ax.set_xlim(cden_lim)
ax.set_ylim(avir_lim)
ax.set_xticklabels('')
ax.set_ylabel(r'$\log\alpha_\mathrm{vir}$')

# Non-rising sources
ax = fig.add_subplot(3,2,4)
for s in sources:
    if not data[s]['rising']:
        ax.plot(data[s]['logSigma'], np.log10(data[s]['avir']),
                marker='o', markersize=ms)
ax.plot(cden_lim, np.zeros(2), 'k--', alpha=0.3)
ax.plot(cden_lim, np.zeros(2) + np.log10(2), 'k--', alpha=0.3)
ax.set_xlim(cden_lim)
ax.set_ylim(avir_lim)
ax.set_xticklabels('')
ax.set_yticklabels('')

# alpha_vir normalized to mean

# Rising sources
ax = fig.add_subplot(3,2,5)
for s in sources:
    if data[s]['rising']:
        ax.plot(data[s]['logSigma'],
                np.log10(data[s]['avir']/data[s]['avir_mean']),
                marker='o', markersize=ms)
ax.set_xlim(cden_lim)
ax.set_ylim(avirrel_lim)
ax.set_ylabel(r'$\log\alpha_\mathrm{vir}/\overline{\alpha}_\mathrm{vir}$')
ax.set_xlabel(r'$\log\Sigma$ [M$_\odot$ pc$^{-2}$]')

# Non-rising sources
ax = fig.add_subplot(3,2,6)
for s in sources:
    if not data[s]['rising']:
        ax.plot(data[s]['logSigma'],
                np.log10(data[s]['avir']/data[s]['avir_mean']),
                marker='o', markersize=ms)
ax.set_xlim(cden_lim)
ax.set_ylim(avirrel_lim)
ax.set_yticklabels('')
ax.set_xlabel(r'$\log\Sigma$ [M$_\odot$ pc$^{-2}$]')

# Adjust spacing
plt.subplots_adjust(right=0.95, top=0.95, wspace=0, hspace=0, left=0.2)

# Save
plt.savefig('m31_12co.pdf')



# Repeat previous steps for 13CO data
fig = plt.figure(2, figsize=(4,5))
fig.clf()

# Observers virial diagram

# Rising
ax = fig.add_subplot(3,2,1)
for s in sources:
    if data[s]['rising13'] == True:
        ax.plot(data[s]['logSigma'], data[s]['logs2r13'],
                marker='o', markersize=ms)
avir1 = np.log10(
    (np.pi/5) * G * 10.**np.array(cden_lim) * u.Msun / u.pc**2 /
    (u.km**2 / u.s**2 / u.pc))
ax.plot(cden_lim, avir1, 'k--', alpha=0.3)
ax.plot(cden_lim, avir1 + np.log10(2), 'k--', alpha=0.3)
ax.set_xlim(cden_lim)
ax.set_ylim(s2r13_lim)
ax.set_xticklabels('')
ax.set_ylabel(r'$\log\sigma^2/R$ [km$^2$ s$^{-2}$ pc$^{-1}$]')
ax.set_title('Rising (N={:d})'.format(nrise13),
             size=12)

# Flat/falling
ax = fig.add_subplot(3,2,2)
for s in sources:
    if data[s]['rising13'] == False:
        ax.plot(data[s]['logSigma'], data[s]['logs2r13'],
                marker='o', markersize=ms)
ax.plot(cden_lim, avir1, 'k--', alpha=0.3)
ax.plot(cden_lim, avir1 + np.log10(2), 'k--', alpha=0.3)
ax.set_xlim(cden_lim)
ax.set_ylim(s2r13_lim)
ax.set_yticklabels('')
ax.set_xticklabels('')
ax.set_title('Flat/falling (N={:d})'.format(ntot13-nrise13), size=12)

# alpha_vir,obs versus sigma

# Rising sources
ax = fig.add_subplot(3,2,3)
for s in sources:
    if data[s]['rising13'] == True:
        ax.plot(data[s]['logSigma'], np.log10(data[s]['avir13']),
                marker='o', markersize=ms)
ax.plot(cden_lim, np.zeros(2), 'k--', alpha=0.3)
ax.plot(cden_lim, np.zeros(2) + np.log10(2), 'k--', alpha=0.3)
ax.set_xlim(cden_lim)
ax.set_ylim(avir13_lim)
ax.set_xticklabels('')
ax.set_ylabel(r'$\log\alpha_\mathrm{vir}$')

# Non-rising sources
ax = fig.add_subplot(3,2,4)
for s in sources:
    if data[s]['rising13'] == False:
        ax.plot(data[s]['logSigma'], np.log10(data[s]['avir13']),
                marker='o', markersize=ms)
ax.plot(cden_lim, np.zeros(2), 'k--', alpha=0.3)
ax.plot(cden_lim, np.zeros(2) + np.log10(2), 'k--', alpha=0.3)
ax.set_xlim(cden_lim)
ax.set_ylim(avir13_lim)
ax.set_xticklabels('')
ax.set_yticklabels('')

# alpha_vir normalized to mean

# Rising sources
ax = fig.add_subplot(3,2,5)
for s in sources:
    if data[s]['rising13'] == True:
        ax.plot(data[s]['logSigma'],
                np.log10(data[s]['avir13']/data[s]['avir13_mean']),
                marker='o', markersize=ms)
ax.set_xlim(cden_lim)
ax.set_ylim(avirrel_lim)
ax.set_ylabel(r'$\log\alpha_\mathrm{vir}/\overline{\alpha}_\mathrm{vir}$')
ax.set_xlabel(r'$\log\Sigma$ [M$_\odot$ pc$^{-2}$]')

# Non-rising sources
ax = fig.add_subplot(3,2,6)
for s in sources:
    if data[s]['rising13'] == False:
        ax.plot(data[s]['logSigma'],
                np.log10(data[s]['avir13']/data[s]['avir13_mean']),
                marker='o', markersize=ms)
ax.set_xlim(cden_lim)
ax.set_ylim(avirrel_lim)
ax.set_yticklabels('')
ax.set_xlabel(r'$\log\Sigma$ [M$_\odot$ pc$^{-2}$]')

# Adjust spacing
plt.subplots_adjust(right=0.95, top=0.95, wspace=0, hspace=0, left=0.2)

# Save
plt.savefig('m31_13co.pdf')


# Comparison of 12 and 13CO data for random subset of clouds
nsamp = 8
sources_both = [s for s in sources if data[s]['rising_match'] != None]
#idx = np.random.choice(len(sources_both), nsamp, replace=False)
idx = np.array([ 1, 2, 8, 12, 13, 19, 20, 23]) # Random set from random.choice
sources_sample = [sources_both[i] for i in idx] 
fig = plt.figure(3, figsize=(4,6))
fig.clf()
avir_comp_lim = [-0.1, 0.2]

# Iterate over examples
for i, s in enumerate(sources_sample):
    ax = fig.add_subplot(nsamp//2,2,i+1)
    ax.plot((data[s]['logSigma']-data[s]['logSigma'][0]) /
            (data[s]['logSigma'][-1]-data[s]['logSigma'][0]),
            np.log10(data[s]['avir']/data[s]['avir_mean']),
            'C0', marker='o', markersize=ms, label=r'$^{12}$CO')
    ax.plot((data[s]['logSigma']-data[s]['logSigma'][0]) /
            (data[s]['logSigma'][-1]-data[s]['logSigma'][0]),
            np.log10(data[s]['avir13']/data[s]['avir13_mean']),
            'C1', marker='o', markersize=ms, label=r'$^{13}$CO')
    if i == 1:
        ax.legend(loc='upper right', prop={'size' : 10})
    ax.text(-0.01, 0.15, s, ha='left')

    # Fix limits and axis labels
    ax.set_xlim([-0.1,1.1])
    ax.set_ylim(avir_comp_lim)
    if i % 2 == 1:
        ax.set_yticklabels('')
    else:
        ax.set_ylabel(r'$\log\alpha_\mathrm{vir}/\overline{\alpha}_\mathrm{vir}$')
    if i < nsamp-2:
        ax.set_xticklabels('')
    else:
        ax.set_xlabel(r'$\log\Sigma_\mathrm{rel}$')

# Adjust spacing
plt.subplots_adjust(left=0.18, right=0.95, top=0.95, hspace=0, wspace=0)

# Save
plt.savefig('12_13_comp.pdf')
