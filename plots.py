import matplotlib.pyplot as pp
import numpy as np

def plot_error_ellipse(logpost, p):
    """Plots the location of the mean and the 1-sigma error ellipse in the
    P-R plane for the given params.

    """

    p = logpost.to_params(p)

    pp.plot(np.exp(p['mu'][0]), np.exp(p['mu'][1]), '.k')

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
    """Plots the location of the mean, the 1-sigma error ellipse and the
    background distribution bounds in the P-R plane for the given
    params.

    """

    p = logpost.to_params(p)

    plot_error_ellipse(logpost, p)
    pp.axvline(np.exp(logpost.lower_left[0]), color='k')
    pp.axvline(np.exp(logpost.upper_right[0]), color='k')
    pp.axhline(np.exp(logpost.lower_left[1]), color='k')
    pp.axhline(np.exp(logpost.upper_right[1]), color='k')

def plot_true_foreground_density(logpost, p):
    """Plots the true (un-selected) foreground density in the P-R plane at
    the indicated parameters.

    """

    p = logpost.to_params(p)
        
    cm = logpost.covariance_matrix(p)
    mu = p['mu']

    xmin = np.min(logpost.candidates['Period'])
    xmax = np.max(logpost.candidates['Period'])
    ymin = np.min(logpost.candidates['Radius'])
    ymax = np.max(logpost.candidates['Radius'])

    XS, YS = np.meshgrid(np.exp(np.linspace(np.log(xmin), np.log(xmax), 250)),
                         np.exp(np.linspace(np.log(ymin), np.log(ymax), 250)))
    ZS = logpost.foreground_density(p, XS.flatten(), YS.flatten()).reshape((250, 250))

    pp.pcolormesh(XS, YS, ZS)
    pp.xscale('log')
    pp.yscale('log')

def eta_earths(logpost, chain):
    r"""Returns an array of :math:`\eta_\oplus` computed at each parameter
    sample in the given chain.

    :math:`\eta_\oplus` is defined as 

    .. math::

      \eta_\oplus \equiv \left. \frac{d N}{d (\ln R) \, d (\ln P)} \right|_{P = 1 \mathrm{ yr}, R = 1 R_\oplus}

    where :math:`N` is the number of planets per star.

    """

    flatchain = chain.reshape((-1, chain.shape[2]))

    eas = np.array([p[0]*logpost.foreground_density(p, np.array([[1.0]]), np.array([[1.0]])) for p in flatchain])

    return eas.squeeze()
