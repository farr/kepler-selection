#!/usr/bin/env python

import os.path as op
from pylab import *

if __name__ == '__main__':
    hackweek_dir = '/Users/farr/Documents/Research/KepHackWeek/data'
    outdir = '/Users/farr/Google Drive/Kepler ExoPop Hack 2015/end2end_occ_calc'
    
    allstars = genfromtxt(op.join(hackweek_dir, 'dr24_stellar.csv'), delimiter=',', names=True)

    non_stype_sel = (allstars['logg'] >= 4.0) & (allstars['dutycycle'] >= 0.33) & (allstars['kepmag'] >= 5) & (allstars['dataspan']*allstars['dutycycle'] > 2*365.25)

    allstars = allstars[non_stype_sel]

    # Stellar types
    gsel = (allstars['teff'] >= 5300) & (allstars['teff'] < 6000)
    ksel = (allstars['teff'] >= 3900) & (allstars['teff'] < 5300)
    msel = (allstars['teff'] >= 2400) & (allstars['teff'] < 3900)

    savetxt(op.join(outdir, 'hack_week_g_stars.csv'), allstars[gsel], delimiter=',', header=','.join(allstars.dtype.names))
    savetxt(op.join(outdir, 'hack_week_k_dwarfs.csv'), allstars[ksel], delimiter=',', header=','.join(allstars.dtype.names))
    savetxt(op.join(outdir, 'hack_week_m_dwarfs.csv'), allstars[msel], delimiter=',', header=','.join(allstars.dtype.names))
