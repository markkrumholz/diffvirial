"""
This script generates plots showing the virial diagrams of
non-singular and singular polytropic solutions. The figures it
produces are figures 1 and 2 of Krumholz, Lada, & Forbrich (2025).
"""

import matplotlib.pyplot as plt
import numpy as np
from polytrope_tools import poly_solve, poly_solve_singular, \
    poly_collapse, poly_collapse_singular

# Range of gamma_p values to explore
gamma_p = np.linspace(0.6, 1, 5)

# Compute singular and non-singular solutions for each case
rns = []
solns = []
rns100 = []
solns100 = []
rs = []
sols = []
for gp in gamma_p:
    print("Computing static solution for gamma_p = {:f}".format(gp))
    r, sol = poly_solve(gp, stop_crit=True, surface_normalize=True)
    rns.append(r)
    solns.append(sol)
    r, sol = poly_solve(gp, chi1=np.log(100), surface_normalize=True)
    rns100.append(r)
    solns100.append(sol)
    r, sol = poly_solve_singular(gp)
    rs.append(r)
    sols.append(sol)

# Compute collapse solutions for non-singular cases
rnsc = []
solnsc = []
rns100c = []
solns100c = []
for i in range(len(gamma_p)):
    print("Computing collapse solution for gamma_p = {:f}".format(gamma_p[i]))
    r, sol = poly_collapse(gamma_p[i], rns[i], solns[i],
                           1.01 / np.sqrt(solns[i]['rho'][0]))
    rnsc.append(r)
    solnsc.append(sol)
    r, sol = poly_collapse(gamma_p[i], rns100[i], solns100[i],
                           1.01 / np.sqrt(solns100[i]['rho'][0]))
    rns100c.append(r)
    solns100c.append(sol)

# Compute collapse solution for SIS at different expansion wave radii
xew = np.array([0.01, 0.1, 0.3])  # Ratio of expansion wave radius to cloud radius
rsc = []
solsc = []
for x in xew:
    print("Computing SIS collapse solution for xew = {:f}".format(x))
    r, sol = poly_collapse_singular(1, 1e-4, 1/x)
    rsc.append(r)
    solsc.append(sol)
    
# Set fonts for plots
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=12)

# Set axis limits
colden_lim = [0.8, 30]
colden_lim2 = [0.8, 300]
avir_lim = [0.5, 30]
s2r_lim = [1, 150]
s2r_lim2 = [1, 1500]

# Create plot window
fig = plt.figure(1, figsize=(4,6))
fig.clf()

# Set colors for lines
cmap = plt.get_cmap('magma_r')
line_colors = cmap(0.85 - 0.7 * (gamma_p-gamma_p[0])/(gamma_p[-1]-gamma_p[0]))
cmap2 = plt.get_cmap('viridis')
line_colors2 = cmap2(0.85 - 0.7 * (xew-xew[0])/(xew[-1]-xew[0]))

# First plot: true alpha_vir vs Sigma
ax = fig.add_subplot(3,1,1)

# Non-singular solutions
for sol, lc in zip(solns, line_colors):
    ax.plot(sol['colden'], sol['avir'], color=lc)

# Non-singular unstable solutions
for sol, lc in zip(solns100, line_colors):
    ax.plot(sol['colden'], sol['avir'], color=lc, ls='-.')

# Singular solutions
for sol, lc in zip(sols, line_colors):
    ax.plot(sol['colden'], sol['avir'], color=lc, ls='--')

# Add virial line
ax.plot(colden_lim, [1,1], color='k', alpha=0.3, lw=4)

# Make legend
p1,=ax.plot([-1], [-1], color=line_colors[0],
         label=r'$\gamma_p = {:3.1f}$'.format(gamma_p[0]))
p2,=ax.plot([-1], [-1], color=line_colors[-1],
         label=r'$\gamma_p = {:3.1f}$'.format(gamma_p[-1]))
ax.legend(handles=[p1,p2], loc='upper left', prop={'size' : 10 })

# Adjust axes
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(colden_lim)
ax.set_ylim(avir_lim)
ax.set_ylabel(r'$\alpha_\mathrm{vir}$')
ax.set_xticklabels([])


# Second plot: observer's alpha_vir vs Sigma
ax = fig.add_subplot(3,1,2)

# Non-singular solutions
for sol, lc in zip(solns, line_colors):
    ax.plot(sol['colden'], sol['avir_obs'], color=lc)

# Non-singular unstable solutions
for sol, lc in zip(solns100, line_colors):
    ax.plot(sol['colden'], sol['avir_obs'], color=lc, ls='-.')

# Singular solutions
for sol, lc in zip(sols, line_colors):
    ax.plot(sol['colden'], sol['avir_obs'], color=lc, ls='--')

# Add virial line
ax.plot(colden_lim, [1,1], color='k', alpha=0.3, lw=4)

# Adjust axes
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(colden_lim)
ax.set_ylim(avir_lim)
ax.set_ylabel(r'$\alpha_\mathrm{vir,obs}$')
ax.set_xticklabels([])


# Third plot: virial diagram
ax = fig.add_subplot(3,1,3)

# Non-singular solutions
for r, sol, lc in zip(rns, solns, line_colors):
    ax.plot(sol['colden'], sol['sigma_proj']**2 / r, color=lc)

# Non-singular unstable solutions
for r, sol, lc in zip(rns100, solns100, line_colors):
    ax.plot(sol['colden'], sol['sigma_proj']**2 / r, color=lc, ls='-.')

# Singular solutions
for r, sol, lc in zip(rs, sols, line_colors):
    ax.plot(sol['colden'], sol['sigma_proj']**2 / r, color=lc, ls='--')

# Add virial line
ax.plot(np.logspace(-2,3),
        np.pi/5 * np.logspace(-2,3),
        color='k', alpha=0.3, lw=4)
    
# Make legend
p1,=ax.plot([-1], [-1], 'k', label='Critical')
p2,=ax.plot([-1], [-1], 'k-.', label=r'$\frac{\rho_c}{\rho_s} = 100$')
p3,=ax.plot([-1], [-1], 'k--', label='Singular')
p4,=ax.plot([-1], [-1], 'k', lw=4, alpha=0.3,
            label=r'$\alpha_\mathrm{vir} = 1$')
ax.legend(handles=[p1,p2,p3,p4], loc='upper left', prop={'size' : 10 })

# Adjust axes
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(colden_lim)
ax.set_ylim(s2r_lim)
ax.set_ylabel(r'$\langle\sigma^2\rangle/R$ (arb.~units)')
ax.set_xlabel(r'$\Sigma$ (arb.~units)')

# Adjust spacing
fig.subplots_adjust(hspace=0, right=0.95, top=0.95, left=0.18)

# Save
fig.savefig('avir_polytrope.pdf')


# Create second plot window
fig = plt.figure(2, figsize=(4,6))
fig.clf()

# First plot: true alpha_vir vs Sigma for collapse solutions
ax = fig.add_subplot(3,1,1)

# Non-singular solutions
for sol, lc in zip(solnsc, line_colors):
    ax.plot(sol['colden'], sol['avir'], color=lc)

# Non-singular unstable solutions
for sol, lc in zip(solns100c, line_colors):
    ax.plot(sol['colden'], sol['avir'], color=lc, ls='-.')

# Singular solutions
for sol, lc in zip(solsc, line_colors2):
    ax.plot(sol['colden'], sol['avir'], color=lc, ls='--')

# Add virial line
ax.plot(colden_lim2, [1,1], color='k', alpha=0.3, lw=4)

# Make legend
p1,=ax.plot([-1], [-1], color=line_colors2[0],
         label=r'SIS, $r_\mathrm{{ew}}/r_s = {:4.2f}$'.format(xew[0]))
p2,=ax.plot([-1], [-1], color=line_colors2[1],
         label=r'SIS, $r_\mathrm{{ew}}/r_s = {:4.2f}$'.format(xew[1]))
p3,=ax.plot([-1], [-1], color=line_colors2[2],
         label=r'SIS, $r_\mathrm{{ew}}/r_s = {:4.2f}$'.format(xew[2]))
ax.legend(handles=[p1,p2,p3], loc='upper left', prop={'size' : 10 })

# Adjust axes
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(colden_lim2)
ax.set_ylim(avir_lim)
ax.set_ylabel(r'$\alpha_\mathrm{vir}$')
ax.set_xticklabels([])


# Second plot: observers alpha_vir vs Sigma for collapse solutions
ax = fig.add_subplot(3,1,2)

# Non-singular solutions
for sol, lc in zip(solnsc, line_colors):
    ax.plot(sol['colden'], sol['avir_obs'], color=lc)

# Non-singular unstable solutions
for sol, lc in zip(solns100c, line_colors):
    ax.plot(sol['colden'], sol['avir_obs'], color=lc, ls='-.')

# Singular solutions
for sol, lc in zip(solsc, line_colors2):
    ax.plot(sol['colden'], sol['avir_obs'], color=lc, ls='--')

# Add virial line
ax.plot(colden_lim2, [1,1], color='k', alpha=0.3, lw=4)

# Adjust axes
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(colden_lim2)
ax.set_ylim(avir_lim)
ax.set_ylabel(r'$\alpha_\mathrm{vir,obs}$')
ax.set_xticklabels([])

# Third plot: virial diagram
ax = fig.add_subplot(3,1,3)

# Non-singular solutions
for r, sol, lc in zip(rnsc, solnsc, line_colors):
    ax.plot(sol['colden'], sol['sigma_proj']**2 / r, color=lc)

# Non-singular unstable solutions
for r, sol, lc in zip(rns100c, solns100c, line_colors):
    ax.plot(sol['colden'], sol['sigma_proj']**2 / r, color=lc, ls='-.')

# Singular solutions
for r, sol, lc in zip(rsc, solsc, line_colors2):
    ax.plot(sol['colden'], sol['sigma_proj']**2 / r, color=lc, ls='--')

# Add virial line
ax.plot(np.logspace(-2,3),
        np.pi/5 * np.logspace(-2,3),
        color='k', alpha=0.3, lw=4)

# Adjust axes
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(colden_lim2)
ax.set_ylim(s2r_lim2)
ax.set_ylabel(r'$\langle\sigma^2\rangle/R$ (arb.~units)')
ax.set_xlabel(r'$\Sigma$ (arb.~units)')

# Adjust spacing
fig.subplots_adjust(hspace=0, right=0.95, top=0.95, left=0.18)

# Save
fig.savefig('avir_polytrope_collapse.pdf')
