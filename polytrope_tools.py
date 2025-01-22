"""
This file contains some utility routines for calculating the structure
and virial parameters of polytropes.
"""

import numpy as np
from scipy.integrate import ode, simpson, cumulative_simpson
from scipy.interpolate import make_interp_spline
from scipy.optimize import root_scalar
from scipy.special import hyp2f1, gamma

######################################################################
def mh_rhs(chi, y, gamma_p):
    """
    Right-hand side of the McKee & Hollimann (1999) polytrope
    equations

    Parameters:
       chi : float
          dimensionless log density contrast, rho = rho_c exp(-chi)
       y : ndarray
          values of the homology variables mu and lambda
       gamma_p : float
          polytropic index for the problem

    Returns:
       yprime : ndarray
          the values of d(mu)/d(chi) and d(lambda)/d(chi)
    """

    # Extract current values
    mu = y[0]
    lam = y[1]

    # Get derivatives -- McKee & Hollimann (1999) equations 20 and 21
    muprime = 4 * np.pi * gamma_p * lam**4 / mu - \
        (mu/2) * (4 - 3 * gamma_p)
    lamprime = gamma_p * lam**2 / mu - (lam/2) * (2 - gamma_p)
    
    # Return
    return np.array([muprime, lamprime])


######################################################################
class mh_supercrit:
    """
    This is a toy class used to implement checks for criticality when
    integrating the polytropic equations.
    """
    def __init__(self, gamma_p):
        self.gamma_p = gamma_p
    def check_supercrit(self, chi, y):
        """
        Function that returns -1 if the values of y = (mu, lambda)
        indicate that the solution has passed the critical density
        contrast; used to halt numerical integration

        Parameters:
           chi : float
              log density contrast
           y : ndarray
              array containing (mu, lambda)

        Returns:
           ret : int
              -1 if the criticality condition has been met, 0
              otherwise
        """
        
        # Extract current values
        mu = y[0]
        lam = y[1]

        # Check condition
        if mu**2 / lam**4 > \
           8 * np.pi * self.gamma_p / (4 - 3*self.gamma_p):
            return -1
        else:
            return 0        

######################################################################
def compute_TW(r, rho, sigma, M, vr = 0, initial = True):
    """
    A routine to evaluate the kinetic and gravitational potential
    energy of a spherical structure from a table of radial values

    Parameters:
       r : array
          Array of radial positions
       rho : array
          Densities at radial positions
       sigma : array
          Velocity dispersions are radial positions
       M : array
          Mass interior to eaceh radial position
       vr : array
          Radial velocity at each radial position
       initial : bool
          If true, assume that a structure of constant density,
          velocity dispersion, etc. exists interior to the innermost
          radial coordinate, and include the contribution from this
          material in the integrals

    Returns:
       T : array
          the total cumulative kinetic energy contained inside each
          radial position
       W : array
          the total cumulative gravitational potential energy inside
          each radial position

    Notes:
       Formally the kinetic and potential energies returned are
       defined by the integrals
       T = int (4 pi r^2) 3/2 rho (sigma^2 + vr^2/3) dr
       and W =  -int 4 pi r^2 rho (G M/r) dr, respectively.
    """

    # Convert to logarithmic coordinate
    logr = np.log(r)

    # Evaluate integrals using Simpson's rule
    if initial:
        # Case with a uniform center
        T = 6 * np.pi * \
            cumulative_simpson(rho * (sigma**2 + (1/3)*vr**2) * r**3,
                               x = logr,
                               initial = M[0] * sigma[0]**2)
        W = -4 * np.pi * \
            cumulative_simpson(rho * M * r**2,
                               x = logr,
                               initial = 3/5 * M[0]**2 / r[0])
    else:
        # Case without a uniform center
        T = 6 * np.pi * \
            cumulative_simpson(rho * (sigma**2 + (1/3)*vr**2) * r**3,
                               x = logr, initial=0)
        W = -4 * np.pi * \
            cumulative_simpson(rho * M * r**2,
                               x = logr, initial=0)
    # Return
    return T, W


######################################################################
def compute_MKEproj(r, rho, sigma, M, vr = None, initial = True):
    """
    A routine to compute the mass and kinetic energy contained with a
    given cylindrical radius projected onto a spherical cloud.

    Parameters:
       r : array
          Array of radial positions
       rho : array
          Densities at radial positions
       sigma : array
          Velocity dispersions are radial positions
       M : array
          Mass interior to eaceh radial position
       vr : array
          Radial velocity at each radial position
       initial : bool
          If true, assume that a structure of constant density,
          velocity dispersion, etc. exists interior to the innermost
          radial coordinate, and include the contribution from this
          material in the integrals

    Returns:
       Mproj : array
          the total mass contained within each cylindrical radius R,
          evaluated at a set of cylindrical radii R = r
       Tproj : array
          the total kinetic energy contained within each cylindrical
          radius R, evaluated at a set of cylindrical radii R = r; if
          vr is non-zero, only the component of the radial velocity
          along the line of sight is included in the total energy

    Notes:
       Formally the quantities returned are defined by the integrals
       M_proj = int_0^r[-1] int_0^{theta_max}
          4 pi r^2 sin(theta) rho dtheta dr
       T_proj = int_0^r[-1] int_0^{theta_max}
          4 pi r^2 sin(theta) rho (sigma^2 + vr^2 cos^2 theta) dtheta dr
       where theta_max = sin^-1 [min(R/r), 1]
    """
    
    # Convert to logarithmic coordinates
    logr = np.log(r)

    # Loop over projected radii
    Mproj = []
    KEproj = []
    for R in r:

        # Get contributions from isotropic part of integral using
        # Simpson's rule
        fA = np.ones(r.shape)
        idx = r > R
        fA[idx] = 1 - np.sqrt(1 - (R/r[idx])**2)
        Mproj.append(simpson(fA * rho * r**3, x = logr))
        KEproj.append(simpson(fA * rho * sigma**2 * r**3, x = logr))
        if initial:
            Mproj[-1] += M[0] / (4*np.pi)
            KEproj[-1] += M[0] * sigma[0]**2 / (4*np.pi)

        # Now add vr part, which involves an extra factor of cos(theta)
        if vr is not None:
            fvr = np.ones(r.shape) / 3
            fvr[idx] *= 1 - (1 - (R/r[idx])**2)**1.5
            KEproj[-1] += simpson(fvr * rho * vr**2 * r**3, x = logr)

    # Multiply by prefactors
    Mproj = np.array(Mproj) * 4 * np.pi
    KEproj = np.array(KEproj) * 2 * np.pi

    # Return
    return Mproj, KEproj
        
######################################################################
def poly_solve(gamma_p, chi0 = 1e-6, chi1 = 10, nsamp = 500,
               stop_crit = False, surface_normalize = False):
    """
    Solver to compute the structure of non-singular polytropes and
    derive various parameter from them using the McKee & Hollimann
    formulation of the polytropic equations

    Parameters:
       gamma_p : float
          polytropic index for the problem
       chi0 : float
          value of chi at which to start numerical integration; should
          be close to zero
       chi1 : float
          value of chi at which to end numerical integration
       nsamp : int
          number of sample points in the output grid
       stop_crit : bool
          if True, stop integration when the critical value of chi for
          stability is reached
       surface_normalize : bool
          if True, normalize the solutions to have rho = 1 and sigma =
          1 at the surface rather than the center

    Returns:
       r : ndarray
          array of nsamp values giving dimensionless radius
       sol : dict
          a dict containing a series of ndarrays giving values of
          various dimensionless parameters at each r; the contents of
          sol are:
          'rho' -- density
          'sigma' -- velocity dispersion
          'P' -- pressure
          'M' -- mass
          'avir' -- virial parameter
          'mu' -- M&H dimensionless mass variable
          'lam' -- M&H dimensionless radius variable
          'colden' -- projected density, treating values of r as
             projected radius
          'sigma_proj' -- mass-weighted mean velocity dispersion
             inside projected radius
          'avir_obs' -- observationally-inferred virial parameter
             inside projected radius
    """

    # Create ode solver for system
    solver = ode(mh_rhs).set_integrator("dop853")

    # Set function to break at critical point if requested
    if stop_crit:
        solver.set_solout( mh_supercrit(gamma_p).check_supercrit )

    # Set initial value of series expansion expression
    lam0 = np.sqrt(3 * gamma_p * chi0 / (2 * np.pi))
    mu0 = (4/3 * np.pi * lam0**3)
    y0 = [mu0, lam0]
    solver.set_initial_value(y0, chi0).set_f_params(gamma_p)

    # Integrate, outputting on a logarithmically-spaced grid
    chi = np.logspace(np.log10(chi0), np.log10(chi1), nsamp)
    res = np.zeros((2, nsamp))
    res[0,0] = mu0
    res[1,0] = lam0
    for i in range(1,nsamp):
        res[:,i] = solver.integrate(chi[i])
        if stop_crit and solver.t != chi[i] or not solver.successful():
            chi = chi[:i]
            res = res[:,:i]
            break
    mu = res[0,:]
    lam = res[1,:]

    # Get density and velocity dispersion at each sample point
    rho = np.exp(-chi)
    sigma = np.exp(-(gamma_p-1) * chi / 2)

    # For each chi, get the corresponding value of r
    r = sigma * lam * np.exp(chi/2)

    # Get pressure
    P = rho * sigma**2

    # Get enclosed mass
    M = mu * sigma**3 / np.sqrt(rho)
    
    # Evaluate alpha_vir
    T, W = compute_TW(r, rho, sigma, M)
    avir = -2 * T / W

    # Evaluate projected mass and kinetic energy and use them to get
    # projected sigma and observed alpha_vir
    Mproj, KEproj = compute_MKEproj(r, rho, sigma, M)
    colden = Mproj / (np.pi * r**2)
    sigma_proj = (2 * KEproj / Mproj)**0.5
    avir_obs = 5 * sigma_proj**2 / (np.pi * r * colden)

    # Renormalize to surface values if requested
    if surface_normalize:
        rho_fac = rho[-1]
        sigma_fac = sigma[-1]
        r_fac = sigma_fac / np.sqrt(rho_fac)
        r = r / r_fac
        rho = rho / rho_fac
        sigma = sigma / sigma_fac
        P = P / (rho_fac * sigma_fac**2)
        M = M / (rho_fac * r_fac**3)
        T = T / (rho_fac * r_fac**3 * sigma_fac**2)
        W = W / (rho_fac**2 * r_fac**5)
        colden = colden / (rho_fac * r_fac)
        sigma_proj = sigma_proj / sigma_fac
    
    # Return
    return r, { 'rho' : rho, 'sigma' : sigma, 'P' : P,
                'M' : M, 'avir' : avir,
                'T' : T, 'W' : W,
                'mu' : mu, 'lam' : lam,
                'colden' : colden, 'sigma_proj' : sigma_proj,
                'avir_obs': avir_obs }

######################################################################
def fA_int(R, vp, k):
    """
    A utility routine to return the result of the integral
    4 pi int_0^R (r/R)^k f_A(vp, r) r^2 dr, where
    f_A = 1 - sqrt(1 - R^2/r^2) for r > R, and 1 for r < R

    Parameters:
       R : float
          The value of R in the formula above
       vp : float or array
          The value of varpi in the formula above
       k : float
          The value of k in the formula above

    Returns:
       fA_int : float or array
          The value of the integral above

    Notes:
       The analytic expression used here was derived using Mathematica
    """
    
    term1 = -np.pi**(3/2) * R**(-k) *  vp**(k+3) * \
        gamma(-3/2 - k/2) / gamma(-k/2)
    term2 = -4/(k+3) * np.pi * R**3 * (-1 +
                                 hyp2f1(-1/2, -(k+3)/2, -(k+1)/2,
                                        (vp/R)**2))
    return term1 + term2


######################################################################
def poly_solve_singular(gamma_p, chi0 = -10, nsamp = 500):
    """
    Solver to compute the structure of singular polytropes and
    derive various parameter from them using the McKee & Hollimann
    formulation of the polytropic equations. Singular polytropes use a
    density normalized to the surface instead of the center (since the
    center is undefined), so chi = ln(rho_s / rho), and the chi
    coordinate runs from -infinity to 0.

    Parameters:
       gamma_p : float
          polytropic index for the problem
       chi0 : float
          innermost value of chi over which to sample
       nsamp : int
          number of sample points in the output grid; distributed
          uniformly from chi0 to 1

    Returns:
       r : ndarray
          array of nsamp values giving dimensionless radius
       sol : dict
          a dict containing a series of ndarrays giving values of
          various dimensionless parameters at each r; the contents of
          sol are:
          'rho' -- density
          'sigma' -- velocity dispersion
          'P' -- pressure
          'M' -- mass
          'avir' -- virial parameter
          'colden' -- projected density, treating values of r as
             projected radius
          'sigma_proj' -- mass-weighted mean velocity dispersion
             inside projected radius
          'avir_obs' -- observationally-inferred virial parameter
             inside projected radius
    """

    # Make sure we have a valid gamma_p
    if gamma_p >= 1.2:
        raise ValueError("projected quantities are finite only for "
                         "gamma_p < 1.2")
    
    # Set up chi grid
    chi = np.linspace(chi0, 0, nsamp)

    # Set powerlaw indices on rho, P, sigma
    krho = -2 / (2 - gamma_p)
    kP = -2 * gamma_p / (2 - gamma_p)
    ks = (1 - gamma_p) / (2 - gamma_p)

    # Get rho, sigma, and P
    rho = np.exp(-chi)
    sigma = np.exp(-(gamma_p-1) * chi / 2)
    P = rho * sigma**2

    # Get r
    lam = np.sqrt( (4 - 3*gamma_p) * gamma_p / (2 * np.pi) ) \
        / (2 - gamma_p)
    r = lam * sigma / np.sqrt(rho)

    # Get M
    mu = np.sqrt( 2 * (4 - 3*gamma_p) * gamma_p**3 / np.pi ) \
        / (2 - gamma_p)**2
    M = mu * sigma**3 / np.sqrt(rho)

    # Get alpha_vir; this makes use of the analytic results derived in
    # equation (27) and (31) of M&H
    psi = (4 - 3*gamma_p) / (6 - 5*gamma_p)
    T = 1.5 * psi * M * sigma**2
    W = -psi * M**2 / r
    avir = -2 * T / W

    # Get column density
    M_proj = fA_int(r[-1], r, krho)
    colden = M_proj / (np.pi * r**2)

    # Get projected sigma
    KE_proj = 0.5 * fA_int(r[-1], r, krho + 2*ks)
    sigma_proj = (2 * KE_proj / M_proj)**0.5

    # Get observers virial ratio
    avir_obs = 5 * sigma_proj**2 / (np.pi * r * colden)    

    # Return
    return r, { 'rho' : rho, 'sigma' : sigma, 'P' : P,
                'M' : M, 'avir' : avir,
                'T' : T, 'W' : W,
                'colden' : colden, 'sigma_proj' : sigma_proj,
                'avir_obs': avir_obs }

######################################################################
def hunter_eq(xi, rhs):
    """
    A utility function to return the residual and its derivative for
    use in inverting the coordinate transformation used in the Hunter
    (1962) solution for pressureless collapse, xi + (1/2) sin (2 xi) =
    pi tau/2

    Parameters:
       xi : float
          A proposed value of xi
       rhs : float
          The value of the RHS, pi tau/2

    Returns:
       resid : float
          The residual xi + (1/2) sin (2 xi) - RHS
       deriv : float
          The derivative of the residual with respect to xi
    """    
    s = np.sin(2*xi)
    c = np.cos(2*xi)
    return (xi + 0.5 * s - rhs,
            1 + c,
            -2 * s)

######################################################################
def poly_collapse(gamma_p, r0, sol, tau):
    """
    Compute a pressureless collapse solution from a hydrostatic
    polytrope initial condition.

    Parameters:
       gamma_p : float
          polytropic index for the problem
       r0 : float
          radius returned by poly_solve or poly_solve_singular, giving
          the initial radius of each shell
       sol : dict
          solution object returned by poly_solve or
          poly_solve_singular
       tau : float
          time expressed in units of the free-fall time at the
          outermost value of r

    Returns:
       r : ndarray
          array of values giving dimensionless radius of each shell at
          time tau
       sol : dict
          a dict containing a series of ndarrays giving values of
          various dimensionless parameters at each r; the contents of
          sol are:
          'rho' -- density
          'sigma' -- velocity dispersion
          'vr' -- radial velocity
          'P' -- pressure
          'M' -- mass
          'avir' -- virial parameter
          'colden' -- projected density, treating values of r as
             projected radius
          'sigma_proj' -- mass-weighted mean velocity dispersion
             inside projected radius
          'avir_proj' -- observationally-inferred virial parameter
             inside projected radius
    """

    # First compute the free-fall time for every shell relative to the
    # outermost one, which we set to unity
    tff = np.sqrt(sol['rho'][-1]/sol['rho'])

    # Create array to hold new radii
    r = np.zeros(r0.shape)

    # Find index of innermost shell that has not fallen to the origin
    # by this time
    idxmin = np.argmax(tff > tau)

    # Find new radii for those shells where we can
    xi = np.pi/4
    for i in range(idxmin, r.size):
        xisol = root_scalar(hunter_eq,
                            args=(np.pi/2 * tau/tff[i],),
                            x0=xi,
                            fprime=True, fprime2=True,
                            method='halley')
        if not xisol.converged:
            xisol = root_scalar(hunter_eq,
                                args=(np.pi/2 * tau/tff[i],),
                                bracket=(0, np.pi/2),
                                method='brentq')
            if not xisol.converged:
                raise ValueError("convergence failure in root "
                                 "finding!")
        xi = xisol.root
        r[i] = r0[i] * np.cos(xi)**2

    # Set shell velocities
    vr = np.ndarray(r.shape)
    vr[:idxmin] = np.nan
    vr[idxmin:] = -np.sqrt(sol['M'][idxmin:] *
                           (1/r[idxmin:] - 1/r0[idxmin:]))

    # Set shell densities; we do this by constructing a B-spline fit
    # to M versus r, then differentiating that
    rho = np.ndarray(r.shape)
    rho[:idxmin] = np.inf
    if idxmin < rho.size:
        bspl = make_interp_spline(r[idxmin:], sol['M'][idxmin:])
        bspl_deriv = bspl.derivative()
        dMdr = bspl_deriv(r[idxmin:])
        rho[idxmin:] = dMdr / (4 * np.pi * r[idxmin:]**2)

    # Get velocity dispersions and pressure from densities
    sigma = sol['sigma'] * (rho / sol['rho'])**((gamma_p-1)/2)
    P = sol['P'] * (rho / sol['rho'])**gamma_p

    # Get virial parameter for gas not yet collapsed to the center;
    # note that this will do strange things once the core has
    # collapsed
    T, W = compute_TW(r[idxmin:], rho[idxmin:], sigma[idxmin:],
                      sol['M'][idxmin:], vr[idxmin:])
    avir = np.ndarray(r.shape)
    avir[:idxmin] = np.nan
    avir[idxmin:] = -2 * T / W

    # Get projected quantities
    Mproj, KEproj = compute_MKEproj(r[idxmin:], rho[idxmin:],
                                    sigma[idxmin:], sol['M'][idxmin:],
                                    vr[idxmin:])
    colden = np.ndarray(r.shape)
    colden[:idxmin] = np.nan
    colden[idxmin:] = Mproj / (np.pi * r[idxmin:]**2)
    sigma_proj = np.ndarray(r.shape)
    sigma_proj[:idxmin] = np.inf
    sigma_proj[idxmin:] = (2 * KEproj / Mproj)**0.5
    avir_obs = np.ndarray(r.shape)
    avir_obs[:idxmin] = np.nan
    avir_obs[idxmin:] = 5 * sigma_proj[idxmin:]**2 / \
                                       (np.pi * r[idxmin:] * colden[idxmin:])

    # Return
    return r, { 'vr' : vr, 'M' : sol['M'],
                'T' : T, 'W' : W,
                'rho' : rho, 'sigma' : sigma, 'P' : P,
                'avir' : avir, 'colden' : colden,
                'sigma_proj' : sigma_proj, 'avir_obs' : avir_obs }

######################################################################
def sps_rhs(x, av, gamma_p):
    """
    Function to define the right hand side for the signular polytropic
    sphere expansion wave ODE, using the formulation from Appendix A
    of McLaughlin & Pudritz (1997)

    Parameters:
       x : float
          dimensionless position
       av : array
          array of two elements (a, v), where a is the dimensionless
          density and v is the dimensionless velocity
       gamma_p : float
          polytropic index

    Returns:
       avprime : array
          array of two elements (da/dx, dv/dx)
    """
    
    a = av[0]
    v = av[1]
    if gamma_p != 1:
        b = (2 - gamma_p) * x - v
        gm = gamma_p - 1
        f = b**2 - a**gm
        aprime = a/f * ( (a / (4-3*gamma_p) - (2/x)*b) * b - gm * (2*b + v) )
        vprime = 1/f * ( (a*b / (4-3*gamma_p) - 2 * a**gm/x) * b
                         - gm * (2*a**gm + v*b) )
    else:
        b = x - v
        bfac = b / (b**2 - 1)
        aprime = a * (a - 2 * b/x) * bfac
        vprime = (a * b - 2 / x) * bfac
    return np.array([aprime, vprime])

######################################################################
def sps_rhs_jac(x, av, gamma_p):
    """
    Function to return the Jacobian of the right hand side of the
    singular polytropic sphere system as implemented in sps_rhs

    Parameters:
       x : float
          dimensionless position
       av : array
          array of two elements (a, v), where a is the dimensionless
          density and v is the dimensionless velocity
       gamma_p : float
          polytropic index

    Returns:
       jac : list
          nested 2 x 2 list, containing the derivatives
          [[d^2a / dx da, d^2a / dx dv],
           [d^2v / dx da, d^2v / dx dv]]
    """

    a = av[0]
    v = av[1]
    b = (2 - gamma_p) * x - v
    agm1 = a**(gamma_p-1)
    agm2 = a**(gamma_p-2)
    d = b**2 - agm1
    g4 = 4 - 3*gamma_p
    gm = gamma_p - 1
    darhs = [
        1/d * (a * b / g4 +
               -gm * (v + 2*b) + b * (a/g4 - 2*b/x) +
               agm1 * gm * (-gm * (v + 2*b) + b * (a/g4 - 2*b/x)) / d ),
        1/d * (a * (gm - a/g4 + 4*b/x) +
               2 * a * b * (-gm * (v + 2*b) + b * (a/g4 - 2*b/x)) / d)
    ]
    dvrhs = [
        1/d * (-2 * gm**2 * agm2 + b * (-2 * gm * agm2/x + b/g4) +
               gm * agm2 * (-gm * (2*agm1 + v*b) +
                            b * (-2*agm1/x + a*b/g4)) / d),
        1/d * (2 * agm1/x - 2 * a * b/g4 - gm * (b-v) +
               2 * b * ( -gm * (2*agm1 + v*b) +
                         b * (-2*agm1/x + a*b/g4) ) / d)
    ]
    return [darhs, dvrhs]
    

######################################################################
def poly_collapse_singular(gamma_p, x0, x1, nsamp = 500):
    """
    Compute a solution for a collapsing singular polytropic sphere

    Parameters:
       gamma_p : float
          polytropic index for the problem
       x0 : float
          inner radius to sample, in units where x = 1 is the position
          of the expansion wave
       x1 : float
          position of the cloud surface, in units where x = 1 is the
          position of the expansion wave
       nsamp : int
          number of sample points from x0 to x1; points are
          logarithmically spaced

    Returns:
       x : ndarray
          positions of sample points
       sol : dict
          a dict containing a series of ndarrays giving values of
          various dimensionless parameters at each x; the contents of
          sol are:
          'rho' -- density
          'sigma' -- velocity dispersion
          'P' -- pressure
          'M' -- mass
          'avir' -- virial parameter
          'colden' -- projected density, treating values of r as
             projected radius
          'sigma_proj' -- mass-weighted mean velocity dispersion
             inside projected radius
          'avir_obs' -- observationally-inferred virial parameter
             inside projected radius
    """

    # Create ode solver for system
    solver = ode(sps_rhs, sps_rhs_jac).set_integrator("vode")
    
    # Set values of sample points; we expand these in the low and high
    # directions if necessary to ensure the validity of asymptotic
    # series expressions used below, and strip off the extra points
    # later if needed
    x = np.logspace(np.log10(x0), np.log10(x1), nsamp)
    if x[0] > 1e-4:
        x = np.append([1e-4], x)
    if x[-1] < 1e4:
        x = np.append(x, [1e4])

    # Set initial values
    CHSE = (2 * (4 - 3*gamma_p) / (2 - gamma_p)**2)**(1/(2-gamma_p))
    yew = [CHSE * x[-1]**(-2/(2-gamma_p)), 0]
    solver.set_initial_value(yew, x[-1]).set_f_params(gamma_p).\
        set_jac_params(gamma_p)

    # Integrate
    res = np.zeros((2,x.size))
    res[:,-1] = yew
    for i in range(x.size-2,-1,-1):
        res[:,i] = solver.integrate(x[i])
        if not solver.successful():
            return x, { 'x' : x, 'rho' : res[0,:], 'vr' : res[1,:] }

    # Truncate outer part of cloud if we added it
    if x[-1] > x1:
        x = x[:-1]
        res = res[:,:-1]

    # Extract a and v, derive m
    a = res[0,:]
    v = res[1,:]
    beta = (2 - gamma_p) * x - v
    m = beta * a * x**2 / (4 - 3*gamma_p)

    # Get scale factors to go from the unit system of the ODEs to a
    # unit system defined by G = 1 and rho = sigma = 1 (and therefore
    # polytropic constant K = 1) at the cloud surface; we use the
    # relations
    # fac = 4 pi G t^2 = alpha / rho
    # at = sqrt( K gamma / (4 pi G t^2)^(1-gamma) )
    fac = a[-1]
    t = np.sqrt(fac / (4*np.pi))
    at = np.sqrt(gamma_p / fac**(1-gamma_p) )

    # Convert to target unit system
    rho = a / fac
    P = rho**gamma_p
    sigma = np.sqrt(P/rho)
    vr = v * at
    M = m * at**3 * t
    r = x * at * t
    
    # Get alpha_vir; note that we use the analytic form of the
    # solution at small radii, which we supply by hand here by
    # integrating the analytic density, pressure, and velocity
    # profiles from 0 to r[0], using the general relation that for any
    # quantity q that obeys q = q0 (r / r0)^-k, the integral of that
    # quantity over volume from 0 to r0 is 4 pi q0 r0^3 / (3 - k)
    T, W = compute_TW(r, rho, sigma, M, vr=vr, initial=False)
    T0k = 4 * (4 - 3*gamma_p) * np.pi * np.sqrt(2 * x[0] * M[0]**3)
    T0p = 1.5 * 4 * np.pi * P[0] * r[0]**3 / (3 - 3*gamma_p/2)
    T0k = 0.5 * 4 * np.pi * rho[0] * vr[0]**2 * r[0]**3 / (3 - 5/2)
    W0 = -4 * np.pi * M[0] * rho[0] * r[0]**3 / (3 - 5/2)
    T += T0p + T0k
    W += W0
    avir = -2 * T / W

    # Evaluate projected mass and kinetic energy; as with the
    # non-projected quantities, we need to add in a correction for the
    # part of the singular structure that is inside the innermost zone
    Mproj, KEproj = compute_MKEproj(r, rho, sigma, M, vr=vr, initial=False)
    Mproj += M[0]
    KEproj += T0p + T0k/3  # Divide kinetic part by 3 due to projection
    
    # Get column density, velocity dispersion, and observed virial
    # ratio
    colden = Mproj / (np.pi * r**2)
    sigma_proj = (2 * KEproj / Mproj)**0.5
    avir_obs = 5 * sigma_proj**2 / (np.pi * r * colden)

    # Truncate the first sample point if we added it
    if x[0] < x0:
        r = r[1:]
        rho = rho[1:]
        sigma = sigma[1:]
        P = P[1:]
        vr = vr[1:]
        M = M[1:]
        Mproj = Mproj[1:]
        KEproj = KEproj[1:]
        avir = avir[1:]
        T = T[1:]
        W = W[1:]
        colden = colden[1:]
        sigma_proj = sigma_proj[1:]
        avir_obs = avir_obs[1:]
        x = x[1:]
    
    # Return
    return r, { 'x' : x, 'rho' : rho, 'sigma' : sigma, 'P' : P, 'vr' : vr,
                'M' : M, 'avir' : avir,
                'T' : T, 'W' : W,
                'rew' : at * t,
                'Mproj' : Mproj, 'KEproj' : KEproj,
                'colden' : colden, 'sigma_proj' : sigma_proj,
                'avir_obs': avir_obs }

    
