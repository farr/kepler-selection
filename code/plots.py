import matplotlib.colors as mc
import matplotlib.pyplot as pp
import numpy as np
import os.path as op
import plotutils.autocorr as ac
import plotutils.plotutils as pu
import scipy.stats as ss
import triangle as tri

def setup():
    fig_width_pt = 469.75502
    inches_per_pt = 1.0/72.27

    fig_size = [fig_width_pt*inches_per_pt,
                fig_width_pt*inches_per_pt]

    params = { 'backend' : 'pdf',
               'axes.labelsize' : 10,
               'font.size' : 10,
               'legend.fontsize' : 10,
               'xtick.labelsize' : 8,
               'ytick.labelsize' : 8,
               'text.usetex' : True,
               'figure.figsize' : fig_size,
               'figure.autolayout' : True,
               'image.cmap' : 'cubehelix'}

    pp.rcParams.update(params)

def pcolormesh(x, y, z, *args, **kwargs):
    z = 0.25*(z[:-1,:-1] + z[:-1, 1:] + z[1:,:-1] + z[1:,1:])

    pp.pcolormesh(x, y, z, *args, **kwargs)

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

    pcolormesh(XS, YS, ZS)
    pp.xscale('log')
    pp.yscale('log')

def par_range(xs):
    """Returns ``(mean(xs), mean(xs)-x5, x95-mean(xs))`` where ``x5`` and
    ``x95`` are the 5-th and 95-th percentile of the ``xs``.

    """

    mu = np.mean(xs)

    return (mu, mu - np.percentile(xs, 5), np.percentile(xs, 95) - mu)

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

    eas = eas.squeeze()

    return eas.reshape(chain.shape[:2])

def correlation_coefficients(logpost, chain):
    """Returns the correlation coefficient between P and R.

    """

    flatchain = chain.reshape((-1, chain.shape[2]))

    rs = []
    for p in flatchain:
        cm = logpost.covariance_matrix(p)
        rs.append(cm[0,1]/np.sqrt(cm[0,0]*cm[1,1]))
    rs = np.array(rs)

    return rs.reshape(chain.shape[:2])

def plot_eta_earth_histogram(eta_earths, outdir=None):
    pu.plot_histogram_posterior(eta_earths.flatten(), normed=True, histtype='step', color='k')
    pp.xlabel(r'$\eta_\oplus$')
    pp.ylabel(r'$p\left(\eta_\oplus\right)$')

    pp.axvline(np.percentile(eta_earths, 5), color='k')
    pp.axvline(np.percentile(eta_earths, 95), color='k')

    if outdir is not None:
        pp.savefig(op.join(outdir, 'eta-earth.pdf'))

def plot_foreground_distributions(logpost, chain, N=1000, outdir=None):
    fchain = chain.reshape((-1, chain.shape[2]))

    dNdR = 0
    dNdP = 0
    dNdPdR = 0
    pbacks = 0

    p0 = logpost.to_params(fchain[0,:])

    cm = logpost.covariance_matrix(p0)

    pmin = p0['mu'][0] - 3.0*cm[0,0]
    pmax = p0['mu'][0] + 3.0*cm[0,0]
    rmin = p0['mu'][1] - 3.0*cm[1,1]
    rmax = p0['mu'][1] + 3.0*cm[1,1]

    pmin = min(pmin, np.min(np.log(logpost.candidates['Period'])))
    pmax = max(pmax, np.max(np.log(logpost.candidates['Period'])))
    rmin = min(rmin, np.min(np.log(logpost.candidates['Radius'])))
    rmax = max(rmax, np.max(np.log(logpost.candidates['Radius'])))

    ps = np.exp(np.linspace(pmin, pmax, 100))
    rs = np.exp(np.linspace(rmin, rmax, 100))

    PS, RS = np.meshgrid(ps, rs)

    for p in fchain:
        cm = logpost.covariance_matrix(p)
        mu = logpost.to_params(p)['mu']

        dNdR += p[0]*ss.norm.pdf(np.log(rs), loc=mu[1], scale=np.sqrt(cm[1,1]))
        dNdP += p[0]*ss.norm.pdf(np.log(ps), loc=mu[0], scale=np.sqrt(cm[0,0]))
        dNdPdR += p[0]*logpost.foreground_density(p, PS.flatten(), RS.flatten()).reshape((100, 100))

        rho_fore = p[0]*logpost.foreground_density(p, logpost.candidates['Period'],
                                                   logpost.candidates['Radius'])
        rho_back = p[1]*logpost.background_density(p, logpost.candidates['Period'],
                                                   logpost.candidates['Radius'])

        pbacks += rho_back / (rho_fore + rho_back)

    dNdR /= fchain.shape[0]
    dNdP /= fchain.shape[0]
    dNdPdR /= fchain.shape[0]
    pbacks /= fchain.shape[0]

    pp.subplot(2,2,1)
    pp.plot(rs, dNdR, '-k')
    pp.axis(xmin=rs[0], xmax=rs[-1])
    pp.xscale('log')
    pp.yscale('log')
    pp.xlabel(r'$R$ ($R_\oplus$)')
    pp.ylabel(r'$dN/d\ln R$')

    pp.subplot(2,2,2)
    pcolormesh(PS, RS, dNdPdR, norm=mc.LogNorm())
    pp.colorbar()
    pp.xlabel(r'$P$ ($\mathrm{yr}$)')
    pp.ylabel(r'$R$ ($R_\oplus$)')
    pp.xscale('log')
    pp.yscale('log')
    pp.axis(xmin=ps[0], xmax=ps[-1], ymin=rs[0], ymax=rs[-1])

    pp.subplot(2,2,3)
    pp.scatter(logpost.candidates['Period'], logpost.candidates['Radius'], c=pbacks, norm=mc.LogNorm())
    pp.colorbar()
    pp.xscale('log')
    pp.yscale('log')
    pp.xlabel(r'$P$ ($\mathrm{yr}$)')
    pp.ylabel(r'$R$ ($R_\oplus$)')
    pp.axis(xmin=ps[0], xmax=ps[-1], ymin=rs[0], ymax=rs[-1])

    pp.subplot(2,2,4)
    pp.plot(ps, dNdP, '-k')
    pp.axis(xmin=ps[0], xmax=ps[-1])
    pp.xscale('log')
    pp.yscale('log')
    pp.xlabel(r'$P$ ($\mathrm{yr}$)')
    pp.ylabel(r'$dN/d\ln P$')

    if outdir is not None:
        pp.savefig(op.join(outdir, 'foreground-dist.pdf'))
        
def plot_selection(logpost, chain, Ndraw=100, outdir=None):
    fchain = chain.reshape((-1, chain.shape[2]))

    all_draws = []
    draws = []
    psel = 0
    for p in fchain[np.random.randint(fchain.shape[0], size=Ndraw), :]:
        dcands = logpost.draw(p, logpost.systems['Kepler_ID'],
                              logpost.systems['Mass'],
                              logpost.systems['Radius'],
                              logpost.systems['SNR0'])
        all_draws.append(dcands)
        draws.append(dcands[np.random.random(size=dcands.shape[0]) < 1.0/Ndraw])

        psel += logpost.ptransit(logpost.candidates['Period'],
                                 logpost.candidates['Stellar_Mass'],
                                 logpost.candidates['Stellar_Radius']) * \
            logpost.pdetect(p, logpost.candidates['Period'],
                            logpost.candidates['Radius'],
                            logpost.candidates['SNR0'])
    psel /= Ndraw

    all_candidates = np.concatenate(all_draws)
    candidates = np.concatenate(draws)

    pp.subplot(2,2,1)
    pu.plot_histogram_posterior(logpost.candidates['Radius'], color='k', normed=True, histtype='step', log=True)
    pu.plot_histogram_posterior(all_candidates['Radius'], color='b', normed=True, histtype='step', log=True)
    pp.yscale('log')
    pp.axis(xmin=min(np.min(candidates['Radius']), np.min(logpost.candidates['Radius'])),
            xmax=max(np.max(candidates['Radius']), np.max(logpost.candidates['Radius'])),
            ymin=1e-4)
    pp.xlabel(r'$R$ ($R_\oplus$)')
    pp.ylabel(r'$p(\ln R)$')

    pp.subplot(2,2,2)
    pp.scatter(logpost.candidates['Period'], logpost.candidates['Radius'], color='k', alpha=0.05)
    pp.scatter(candidates['Period'], candidates['Radius'], color='b', alpha=0.05)
    pp.axis(xmin=min(np.min(candidates['Period']), np.min(logpost.candidates['Period'])),
            xmax=max(np.max(candidates['Period']), np.max(logpost.candidates['Period'])),
            ymin=min(np.min(candidates['Radius']), np.min(logpost.candidates['Radius'])),
            ymax=max(np.max(candidates['Radius']), np.max(logpost.candidates['Radius'])))
    pp.ylabel(r'$R$ ($R_\oplus$)')
    pp.xlabel(r'$P$ ($\mathrm{yr}$)')
    pp.xscale('log')
    pp.yscale('log')

    pp.subplot(2,2,3)
    pp.scatter(logpost.candidates['Period'], logpost.candidates['Radius'], c=psel,
               norm=mc.LogNorm())
    pp.axis(xmin=min(np.min(candidates['Period']), np.min(logpost.candidates['Period'])),
            xmax=max(np.max(candidates['Period']), np.max(logpost.candidates['Period'])),
            ymin=min(np.min(candidates['Radius']), np.min(logpost.candidates['Radius'])),
            ymax=max(np.max(candidates['Radius']), np.max(logpost.candidates['Radius'])))
    pp.xscale('log')
    pp.yscale('log')
    pp.ylabel(r'$R$ ($R_\oplus$)')
    pp.xlabel(r'$P$ ($\mathrm{yr}$)')
    pp.colorbar()

    pp.subplot(2,2,4)
    pu.plot_histogram_posterior(logpost.candidates['Period'], color='k', normed=True, histtype='step', log=True)
    pu.plot_histogram_posterior(all_candidates['Period'], color='b', normed=True, histtype='step', log=True)
    pp.yscale('log')
    pp.axis(xmin=min(np.min(candidates['Period']), np.min(logpost.candidates['Period'])),
            xmax=max(np.max(candidates['Period']), np.max(logpost.candidates['Period'])),
            ymin=1e-4)
    pp.xlabel(r'$P$ ($\mathrm{yr}$)')
    pp.ylabel(r'$p(\ln P)$')

    if outdir is not None:
        pp.savefig(op.join(outdir, 'selection.pdf'))
    
def plot_nplanets_histogram(chain, outdir=None):
    tau = ac.autocorrelation_length_estimate(np.mean(chain[:,:,0], axis=0))

    if tau is None:
        nps = chain[:,:,0].flatten()
    else:
        nps = chain[:,::tau,0].flatten()

    pu.plot_histogram_posterior(nps, normed=True, color='k', histtype='step')
    pp.xlabel(r'$R_\mathrm{pl}$')
    pp.ylabel(r'$p(R_\mathrm{pl})$')

    pp.axvline(np.percentile(nps, 5), color='k')
    pp.axvline(np.percentile(nps, 95), color='k')

    if outdir is not None:
        pp.savefig(op.join(outdir, 'npl.pdf'))

def paper_plots(logpost, chain, eta_earths, outdir):
    pp.figure()
    plot_nplanets_histogram(chain, outdir=outdir)

    pp.figure()
    plot_eta_earth_histogram(eta_earths, outdir=outdir)

    pp.figure()
    plot_foreground_distributions(logpost, chain, outdir=outdir)

    pp.figure()
    plot_selection(logpost, chain, outdir=outdir)

def paper_tex(logpost, chain, eta_earths, rs, outdir):
    ea = par_range(100.0*eta_earths)
    rpl = par_range(chain[:,:,0])
    rpk = par_range(np.exp(chain[:,:,3]))
    rc = par_range(rs)
    fp = par_range(100.0*logpost.systems.shape[0]*chain[:,:,1]/logpost.candidates.shape[0])
    pp = par_range(np.exp(chain[:,:,2]))

    with open(op.join(outdir, 'quantities.tex'), 'w') as out:
        out.write('\\newcommand{{\\earange}}{{{0:0.1f}_{{-{1:0.1f}}}^{{+{2:0.1f}}}\\%}}\n'.format(ea[0], ea[1], ea[2]))

        out.write('\\newcommand{{\\rplrange}}{{{0:0.2f}_{{-{1:0.2f}}}^{{+{2:0.2f}}}}}\n'.format(rpl[0], rpl[1], rpl[2]))

        out.write('\\newcommand{{\\rpeakrange}}{{{0:0.2f}_{{-{1:0.2f}}}^{{+{2:0.2f}}}}}\n'.format(rpk[0], rpk[1], rpk[2]))

        out.write('\\newcommand{{\\corrcoeffrange}}{{{0:0.3f}_{{-{1:0.3f}}}^{{+{2:0.3f}}}}}\n'.format(rc[0], rc[1], rc[2]))

        out.write('\\newcommand{{\\fposrange}}{{{0:0.1f}_{{-{1:0.1f}}}^{{+{2:0.1f}}}\\%}}\n'.format(fp[0], fp[1], fp[2]))

        out.write('\\newcommand{{\\ppeakrange}}{{{0:0.3f}_{{-{1:0.3f}}}^{{+{2:0.3f}}}}}\n'.format(pp[0], pp[1], pp[2]))
        
