import matplotlib.pyplot as pp
import numpy as np

def plot_error_ellipse(logpost, p):
    pp.figure()
    
    p = logpost.to_params(p)

    pp.plot(np.exp(p['mu'][0]), np.exp(p['mu'][1]), '.k')

    print p['mu']

    thetas = np.linspace(0, 2*np.pi, 1000)

    cm = logpost.covariance_matrix(p)
    evals, evecs = np.linalg.eig(cm)

    xv = np.sqrt(evals[0])*evecs[:,0]
    yv = np.sqrt(evals[1])*evecs[:,1]

    ct = np.cos(thetas).reshape(-1, 1)
    st = np.sin(thetas).reshape(-1, 1)

    pts = p['mu'] + ct*xv + st*yv

    pp.plot(np.exp(pts[:,0]), np.exp(pts[:,1]), '-k')

    pp.xscale('log')
    pp.yscale('log')

def plot_error_ellipse_and_bounds(logpost, p):
    p = logpost.to_params(p)

    plot_error_ellipse(logpost, p)
    pp.axvline(np.exp(p['lower_left'][0]), color='k')
    pp.axvline(np.exp(p['upper_right'][0]), color='k')
    pp.axhline(np.exp(p['lower_left'][1]), color='k')
    pp.axhline(np.exp(p['upper_right'][1]), color='k')

    pp.loglog(np.exp(logpost.pts[:,0]), np.exp(logpost.pts[:,1]), ',k')
