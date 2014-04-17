import matplotlib.colors as mc
import matplotlib.pyplot as pp
import numpy as np
import os.path as op
import plotutils.autocorr as ac
import plotutils.plotutils as pu
import scipy.stats as ss
import triangle as tri

def emulateapj_setup():
    fig_width_pt = 245.26653
    inches_per_pt = 1.0/72.27

    fig_size = [fig_width_pt*inches_per_pt,
                fig_width_pt*inches_per_pt]

    params = { 'backend' : 'pdf',
               'axes.labelsize' : 10,
               'text.fontsize' : 10,
               'legend.fontsize' : 10,
               'xtick.labelsize' : 8,
               'ytick.labelsize' : 8,
               'text.usetex' : True,
               'figure.figsize' : fig_size,
               'figure.autolayout' : True}

    pp.rcParams.update(params)

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

    eas = eas.squeeze()

    return eas.reshape(chain.shape[:2])

def plot_eta_earth_histogram(eta_earths, outdir=None):
    tau = ac.autocorrelation_length_estimate(np.mean(eta_earths, axis=0))

    thin_eta_earths = eta_earths[:, ::int(round(tau))]

    pu.plot_histogram_posterior(thin_eta_earths.flatten(), normed=True, histtype='step', color='k')
    pp.xlabel(r'$\eta_\oplus$')
    pp.ylabel(r'$p\left(\eta_\oplus\right)$')

    pp.axvline(np.percentile(eta_earths, 5), color='k')
    pp.axvline(np.percentile(eta_earths, 95), color='k')

    if outdir is not None:
        pp.savefig(op.join(outdir, 'eta-earth.pdf'))

def plot_foreground_distributions(logpost, chain, N=1000, outdir=None):
    tau = np.max(ac.emcee_chain_autocorrelation_lengths(chain))

    fchain = chain.reshape((-1, chain.shape[2]))

    step = int(round(float(fchain.shape[0])/N))
    tfchain = fchain[::step,:]

    dNdR = 0
    dNdP = 0
    dNdPdR = 0
    pfores = 0

    p0 = logpost.to_params(tfchain[0,:])

    cm = logpost.covariance_matrix(p0)

    pmin = p0['mu'][0] - 3.0*cm[0,0]
    pmax = p0['mu'][0] + 3.0*cm[0,0]
    rmin = p0['mu'][1] - 3.0*cm[1,1]
    rmax = p0['mu'][1] + 3.0*cm[1,1]

    pmin = min(pmin, np.min(np.log(logpost.candidates['Period'])))
    pmax = max(pmax, np.max(np.log(logpost.candidates['Period'])))
    rmin = min(rmin, np.min(np.log(logpost.candidates['Radius'])))
    rmax = max(rmax, np.max(np.log(logpost.candidates['Radius'])))

    ps = np.exp(np.linspace(pmin, pmax, 250))
    rs = np.exp(np.linspace(rmin, rmax, 250))

    PS, RS = np.meshgrid(ps, rs)

    for p in tfchain:
        cm = logpost.covariance_matrix(p)
        mu = logpost.to_params(p)['mu']

        dNdR += p[0]*ss.norm.pdf(np.log(rs), loc=mu[1], scale=cm[1,1])
        dNdP += p[0]*ss.norm.pdf(np.log(ps), loc=mu[0], scale=cm[0,0])
        dNdPdR += p[0]*logpost.foreground_density(p, PS.flatten(), RS.flatten()).reshape((250, 250))

        rho_fore = p[0]*logpost.foreground_density(p, logpost.candidates['Period'],
                                                   logpost.candidates['Radius'])
        rho_back = p[1]*logpost.background_density(p, logpost.candidates['Period'],
                                                   logpost.candidates['Radius'])

        pfores += rho_fore / (rho_fore + rho_back)

    dNdR /= tfchain.shape[0]
    dNdP /= tfchain.shape[0]
    dNdPdR /= tfchain.shape[0]
    pfores /= tfchain.shape[0]

    pp.subplot(2,1,1)
    pp.plot(rs, dNdR, '-k')
    pp.xscale('log')
    pp.xlabel(r'$R$ ($R_\oplus$)')
    pp.ylabel(r'$p(\ln R)$')

    pp.subplot(2,1,2)
    pp.scatter(logpost.candidates['Period'], logpost.candidates['Radius'], c=pfores)
    pp.colorbar()
    pp.xscale('log')
    pp.yscale('log')
    pp.xlabel(r'$P$ ($\mathrm{yr}$)')
    pp.ylabel(r'$R$ ($R_\oplus$)')
    pp.axis(xmin=ps[0], xmax=ps[-1], ymin=rs[0], ymax=rs[-1])

    if outdir is not None:
        pp.savefig(op.join(outdir, 'foreground-dist-1.pdf'))

    pp.figure()

    pp.subplot(2,1,1)
    pp.pcolormesh(PS, RS, dNdPdR)
    pp.colorbar()
    pp.xlabel(r'$P$ ($\mathrm{yr}$)')
    pp.ylabel(r'$R$ ($R_\oplus$)')
    pp.xscale('log')
    pp.yscale('log')
    pp.axis(xmin=ps[0], xmax=ps[-1], ymin=rs[0], ymax=rs[-1])

    pp.subplot(2,1,2)
    pp.plot(ps, dNdP, '-k')
    pp.xscale('log')
    pp.xlabel(r'$P$ ($\mathrm{yr}$)')
    pp.ylabel(r'$p(\ln P)$')

    if outdir is not None:
        pp.savefig(op.join(outdir, 'foreground-dist-2.pdf'))
        
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

    pp.subplot(2,1,1)
    pu.plot_histogram_posterior(logpost.candidates['Radius'], color='k', normed=True, histtype='step', log=True)
    pu.plot_histogram_posterior(all_candidates['Radius'], color='b', normed=True, histtype='step', log=True)
    pp.xlabel(r'$R$ ($R_\oplus$)')
    pp.ylabel(r'$p(\ln R)$')

    pp.subplot(2,1,2)
    pp.scatter(logpost.candidates['Period'], logpost.candidates['Radius'], c=psel,
               norm=mc.LogNorm())
    pp.axis(xmin=np.min(logpost.candidates['Period']),
            xmax=np.max(logpost.candidates['Period']),
            ymin=np.min(logpost.candidates['Radius']),
            ymax=np.max(logpost.candidates['Radius']))
    pp.xscale('log')
    pp.yscale('log')
    pp.ylabel(r'$R$ ($R_\oplus$)')
    pp.xlabel(r'$P$ ($\mathrm{yr}$)')
    pp.colorbar()

    if outdir is not None:
        pp.savefig(op.join(outdir, 'selection-1.pdf'))

    pp.figure()

    pp.subplot(2,1,1)
    pp.scatter(logpost.candidates['Period'], logpost.candidates['Radius'], color='k', alpha=0.1)
    pp.scatter(candidates['Period'], candidates['Radius'], color='b', alpha=0.2)
    pp.axis(xmin=min(np.min(candidates['Period']), np.min(logpost.candidates['Period'])),
            xmax=max(np.max(candidates['Period']), np.max(logpost.candidates['Period'])),
            ymin=min(np.min(candidates['Radius']), np.min(logpost.candidates['Radius'])),
            ymax=max(np.max(candidates['Radius']), np.max(logpost.candidates['Radius'])))
    pp.ylabel(r'$R$ ($R_\oplus$)')
    pp.xlabel(r'$P$ ($\mathrm{yr}$)')
    pp.xscale('log')
    pp.yscale('log')

    pp.subplot(2,1,2)
    pu.plot_histogram_posterior(logpost.candidates['Period'], color='k', normed=True, histtype='step', log=True)
    pu.plot_histogram_posterior(all_candidates['Period'], color='b', normed=True, histtype='step', log=True)
    pp.xlabel(r'$P$ ($\mathrm{yr}$)')
    pp.ylabel(r'$p(\ln P)$')

    if outdir is not None:
        pp.savefig(op.join(outdir, 'selection-2.pdf'))
    
def plot_nplanets_histogram(chain, outdir=None):
    tau = ac.autocorrelation_length_estimate(np.mean(chain[:,:,0], axis=0))

    nps = chain[:,::tau,0].flatten()

    pu.plot_histogram_posterior(nps, normed=True, color='k', histtype='step')
    pp.xlabel(r'$R$')
    pp.ylabel(r'$p(R)$')

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
