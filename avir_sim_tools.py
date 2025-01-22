"""
Functions to do contour virial analysis on simulation data
"""

import numpy as np
from skimage.segmentation import flood

######################################################################
def project_quantities(d):
    """
    Given a simulation data set consisting of 3d data, generate
    projected quantities
    
    Parameters:
       d : dict
          a dict containing the data set; it must contain the keys
          rho, vx, vy, and vz, which in turn hold 3d arrays containing
          density and 3 components of velocity for each cell

    Returns:
       Nothing

    Notes:
       On return d will have the added keys cden, vcen, and sigma2;
       each of these is a dict with keys 'x', 'y', 'z', giving the
       column density, centroid velocity, and velocity dispersion
       (thermal plus non-thermal) projected in each cardinal direction
    """

    # Get column density and mean velocity projected in each direction
    d['cden'] = { 'x' : np.mean( d['rho'], axis=0 ),
                  'y' : np.mean( d['rho'], axis=1 ),
                  'z' : np.mean( d['rho'], axis=2 ) }
    d['vcen'] = {
        'x' : np.mean( d['rho'] * d['vx'], axis=0 ) / d['cden']['x'],
        'y' : np.mean( d['rho'] * d['vy'], axis=1 ) / d['cden']['y'],
        'z' : np.mean( d['rho'] * d['vz'], axis=2 ) / d['cden']['z']
    }

    # Now get thermal plus non-thermal velocity dispersion in every
    # direction; this requires some care with indices so that we
    # subtract off the means properly
    d['sigma2'] = {
        'x' : 1 + np.mean( d['rho'] * (d['vx'] - d['vcen']['x'])**2,
                           axis=0 ) / d['cden']['x']
    }
    v = d['vy']
    for j in range(d['rho'].shape[1]):
        v[:,j,:] -= d['vcen']['y']
    d['sigma2']['y'] = 1 + np.mean(d['rho'] * v**2, axis=1) / \
        d['cden']['y']
    v = d['vz']
    for k in range(d['rho'].shape[2]):
        v[:,:,k] -= d['vcen']['z']
    d['sigma2']['z'] = 1 + np.mean(d['rho'] * v**2, axis=2) / \
        d['cden']['z']

    # Return
    return

######################################################################
def contour_analysis(d, clev, G, min_pix=0):
    """
    Given a simulation data set and a set of contour levels, find
    contours and compute virial parameters in each contour for each
    projection direction

    Parameters:
       d : dict
          a dict containing the data set; it must contain the keys
          rho, vx, vy, and vz, which in turn hold 3d arrays containing
          density and 3 components of velocity for each cell
       clev : arraylike
          log column density contour levels at which to analyze
       G : float
          numerical value of G for the simulation, which is assumed to
          be in a unit system where mean density = box length = sound
          speed = 1
       min_pix : int
          minimum number of pixels in a region

    Returns:
       cdat : dict
          A list containing the contour data for each projection
          direction. The dict contains three keys, 'x', 'y', and 'z',
          for the three directions. Each dict entry is a list giving the
          data at each contour level, and each entry in the list is a
          dict containing the keys 'npix', 'Reff', 'cden', 'sigma2',
          and 'avir', which give, respectively, the pixel count,
          effective radius, mean column density, mass-weighted mean
          square velocity dispersion, and virial parameter of each
          region identified at that contour level. Note that these
          lists may be empty if no regions exist at a given contour
          level.
    """

    # Loop over projection direction
    cdat = { }
    for pr in ['x', 'y', 'z']:

        # Tile the data in a 3x3 grid to handle periodicity
        sh = d['cden'][pr].shape
        cden_tile = np.tile(d['cden'][pr], (3,3))
        sigma2_tile = np.tile(d['sigma2'][pr], (3,3))

        # Loop over contour levels
        contour_data = []
        for cl in clev:

            # Generate pixel mask at threshold level
            mask = cden_tile >= 10.**cl

            # Now we need ot identify connected pixel regions; we do
            # this by progressively flood-filling: we find a
            # non-masked pixel, do a flood-fill to find a connected
            # region, mask that connected region, and repeat until
            # there are no non-masked regions
            region_data = { 'npix' : [],
                            'cden' : [],
                            'Reff' : [],
                            'sigma2' : [],
                            'avir' : [] }
            while np.amax(mask):

                # Find a non-masked pixel
                idx = np.unravel_index(np.argmax(mask), mask.shape)

                # Flood the region to find all connected pixels above
                # the threshold
                reg_mask = flood(mask, idx)

                # Mask the region we just found
                mask[reg_mask] = False

                # Now begin checks as to whether we should keep this
                # region in our list of contours to analyze. First,
                # check if the region is smaller than our min pixel
                # number, and if so discard.
                npix = np.sum(reg_mask)
                if npix < min_pix:
                    continue

                # Next only keep regions if their centroid falls
                # within the range of real (as opposed to image)
                # pixels, so that we do not double-count regions that
                # appear multiple times because they cross the
                # periodic boundary
                xwgt = np.sum(reg_mask, axis=1)
                ywgt = np.sum(reg_mask, axis=0)
                xcen = np.sum(xwgt * np.arange(cden_tile.shape[0])) \
                    / np.sum(xwgt)
                ycen = np.sum(ywgt * np.arange(cden_tile.shape[1])) \
                    / np.sum(ywgt)
                if xcen < sh[0] or xcen > 2*sh[0] - 1 or \
                   ycen < sh[1] or ycen > 2*sh[1] - 1:
                    continue

                # Now do check (2): make sure  the region we have
                # found does not go all the way to the edge of the
                # tiled box, since if it does this indicates it is a
                # region that extends all the way across the periodic
                if np.amax(reg_mask[0,:]) or \
                   np.amax(reg_mask[-1,:]) or \
                   np.amax(reg_mask[:,0]) or \
                   np.amax(reg_mask[:,-1]):
                    continue

                # If we are here, we have passed all checks, and
                # therefore we should keep this region; get its key
                # statistics: effective radius, mean column density,
                # mass-weighted mean velocity dispersion, virial
                # parameter
                Reff = np.sqrt(npix / (np.pi * sh[0] * sh[1]))
                cden = np.sum(reg_mask * cden_tile) / npix
                sigma2 = np.sum(reg_mask * cden_tile * sigma2_tile) \
                    / (cden * npix)
                avir = 5 * sigma2 / (np.pi * G * Reff * cden)

                # Save
                region_data['npix'].append(npix)
                region_data['Reff'].append(Reff)
                region_data['cden'].append(cden)
                region_data['sigma2'].append(sigma2)
                region_data['avir'].append(avir)

            # Save region data for this contour level
            contour_data.append(region_data)

        # Save contour data
        cdat[pr] = contour_data

    # Return data for all directions
    return cdat


######################################################################
def contour_summary(cdat, group_proj=False, ncmin=1):
    """
    Given a set of contour values returned by contour analysis,
    compute summary statistics on those contours

    Parameters:
       cdat : dict
          A set of contour data in the format returned by
          contour_analysis
       group_proj : bool
          If True, all projection directions are grouped for the
          analysis; if False, they are kept separate
       ncmin : int
          Statistics will only be computed for bins that contain more
          than ncmin contours

    Returns:
       csum : dict
          A set of summary statistics for the contours; if group_proj
          is False, this is a dict with keys 'x', 'y', and 'z', and
          each dict entry is itself a dict containing the keys
          'nreg', 'avir_mean', 'avir_lo', 'avir_med', and 'avir_hi',
          's2r_mean', 's2r_lo', 's2r_med', and 's2r_hi'. These
          contain arrays giving for each contour level, respectively,
          the number of regions, the mass-weighted mean, median, and
          16th, and 84th percentile virial parameters, and the
          mass-weighted mean value of sigma^2 / Reff. If group_proj is
          true, csum is a single dict containing these quantities
          combining contours from all projection directions
    """

    # If we are combining projection directions, first concatenate the
    # data from the different projection directions
    if group_proj:
        nlev = len(cdat['x'])
        cdat_comb = []
        for i in range(nlev):
            lev_data = { 'npix' : [],
                         'cden' : [],
                         'sigma2' : [],
                         'Reff' : [],
                         'avir' : [] }
            for pr in ['x', 'y', 'z']:
                lev_data['npix'] += cdat[pr][i]['npix']
                lev_data['cden'] += cdat[pr][i]['cden']
                lev_data['sigma2'] += cdat[pr][i]['sigma2']
                lev_data['Reff'] += cdat[pr][i]['Reff']
                lev_data['avir'] += cdat[pr][i]['avir']
            cdat_comb.append(lev_data)
        cdat_dummy = { 'all' : cdat_comb }  # Store in dummy projection
    else:
        cdat_dummy = cdat   # Just process original data

    # Now process data
    csum = { }
    for pr in cdat_dummy.keys():

        # Make output holder for this projection
        csum[pr] = { 'nc' : [],
                     'avir_mean' : [],
                     'avir_lo' : [],
                     'avir_med' : [],
                     'avir_hi' : [],
                     's2r_mean' : [],
                     's2r_lo' : [],
                     's2r_med' : [],
                     's2r_hi' : []
                    }

        # Loop over contour levels
        for lev in cdat_dummy[pr]:

            # Handle case of too few contours
            nc = len(lev['npix'])
            if nc < ncmin:
                for k in csum[pr].keys():
                    csum[pr][k].append(np.nan)
                csum[pr]['nc'][-1] = 1
                continue

            # Compute statistics at this contour level
            csum[pr]['nc'].append(nc)
            mass = np.array(lev['cden']) * np.array(lev['npix'])
            csum[pr]['avir_mean'].append(
                np.sum(mass * np.array(lev['avir'])) / np.sum(mass) )
            csum[pr]['s2r_mean'].append(
                np.sum(mass * np.array(lev['sigma2']) /
                       np.array(lev['Reff'])) /
                np.sum(mass) )
            pct = np.percentile(lev['avir'], [15.87, 50, 84.13],
                                weights=mass,
                                method='inverted_cdf')
            csum[pr]['avir_lo'].append(pct[0])
            csum[pr]['avir_med'].append(pct[1])
            csum[pr]['avir_hi'].append(pct[2])
            pct = np.percentile(
                np.array(lev['sigma2']) / np.array(lev['Reff']),
                [15.87, 50, 84.13],
                weights=mass,
                method='inverted_cdf')
            csum[pr]['s2r_lo'].append(pct[0])
            csum[pr]['s2r_med'].append(pct[1])
            csum[pr]['s2r_hi'].append(pct[2])

    # If we are grouping projection, get rid of the dummy dict key
    if group_proj:
        csum = csum['all']

    # Return
    return csum
