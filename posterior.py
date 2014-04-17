import bz2
import glob
import numpy as np
import os.path as op
import scipy.integrate as si
import scipy.misc as sm
import scipy.special as ss
import scipy.stats as st

class Posterior(object):
    """Class representing a posterior object for the selection-effects
    model of the observed Kepler planet number density in the
    period-radius plane.

    """

    def __init__(self, candidates, systems, lower_left=None, upper_right=None):
        r"""Initialise the posterior object.  

        :param candidates: An array with the columns 

           * ``Kepler_ID``
           * ``Period``, period in years of candidate.
           * ``Radius``, radius in Earth radii of candidate.
           * ``Stellar_Radius``, radius in Solar radii of host star.
           * ``Stellar_Mass``, mass in Solar masses of host star.
           * ``SNR0``, SNR of 1 REarth planet in orbit at 1 yr.

          giving the properties of the identified planetary
          candidates.

        :param systems: An array with the columns

           * ``Kepler_ID``
           * ``Radius`` giving the radius in Solar radii
           * ``Mass`` giving the mass in Solar masses
           * ``SNR0`` giving the SNR of a 1 REarth planet in orbit at 1 yr.

          giving the properties of the stars in the observing program
          involved in the search.

        :param lower_left: The lower left boundary of the background
          distribution in the :math:`\ln(P)`-:math:`\ln(R)` plane.  If
          ``None``, then the most extreme candidate point will be
          used.

        :param upper_right: See ``lower_left``.

        """
        self.candidates = candidates
        self.systems = systems
        self.pts = np.log(np.column_stack((candidates['Period'], candidates['Radius'])))

        if lower_left is None:
            self.lower_left = np.min(self.pts, axis=0)
        else:
            self.lower_left = lower_left

        if upper_right is None:
            self.upper_right = np.max(self.pts, axis=0)
        else:
            self.upper_right = upper_right

    @property
    def dtype(self):
        """Gives the numpy datatype of the parameters.

        """

        return np.dtype([('R', np.float),
                         ('Rb', np.float),
                         ('mu', np.float, 2),
                         ('sigma', np.float, 2),
                         ('theta', np.float),
                         ('log_snr_min', np.float),
                         ('log_snr_max', np.float),
                         ('gamma', np.float, 2),
                         ('Pmin', np.float)])

    @property
    def pnames(self):
        """Gives LaTeX names for the parameters.

        """

        return [r'$R$', r'$R_b$',
                r'$\mu_P$', r'$\mu_R$',
                r'$\sigma_1$', r'$\sigma_2$',
                r'$\theta$',
                r'$\log\left(\rho_\mathrm{min}\right)$',
                r'$\log\left(\rho_\mathrm{max}\right)$',
                r'$\gamma_P$', r'$\gamma_R$', r'$P_\mathrm{min}$']

    @property
    def nparams(self):
        """The number of parameters."""
        return 12

    def to_params(self, p):
        """Converts the array ``p`` to a named-array representing parameters.

        """
        return p.view(self.dtype).squeeze()

    def covariance_matrix(self, p, inv=False):
        """Returns the covariance matrix (or inverse, if ``inv=True``)
        corresponding to the parameters ``p``.

        """
        p = self.to_params(p)

        if inv:
            d = np.diag(1.0/(p['sigma']*p['sigma']))
        else:
            d = np.diag(p['sigma']*p['sigma'])

        ct = np.cos(p['theta'])
        st = np.sin(p['theta'])

        r = np.array([[ct, -st], [st, ct]])

        return np.dot(r, np.dot(d, r.T))

    def foreground_density(self, p, ps, rs):
        """Returns the true foreground density for the given parameters and
        planet periods and radii (in years and Earth radii, respectively).

        """
        p = self.to_params(p)

        pts = np.log(np.column_stack((ps, rs)))

        cm = self.covariance_matrix(p, inv=True)

        xs = pts - p['mu']

        return 1.0/(2.0*np.pi*np.prod(p['sigma']))*np.exp(-0.5*np.sum(xs*np.dot(cm, xs.T).T, axis=1))

    def background_density(self, p, ps, rs):
        """Return the background density at the given points in the
        period-radius plane.

        """

        p = self.to_params(p)

        ps = np.atleast_1d(ps)
        rs = np.atleast_1d(rs)

        pts = np.log(np.column_stack((ps, rs)))

        lower_left = self.lower_left.copy()
        lower_left[0] = np.log(p['Pmin'])
        
        center = 0.5*(lower_left + self.upper_right)
        dx = self.upper_right - lower_left
        V = np.prod(dx)
        
        rhos = 1.0/V*(1.0 + np.dot(p['gamma'], (pts - center).T))

        sel = (pts < lower_left) | (pts > self.upper_right)
        sel = sel[:,0] | sel[:,1]

        rhos[sel] = 0.0
        return rhos

    def log_snr(self, ps, rs, snr0s):
        """Returns the log of the SNR for the given periods and radii around
        stars with the given SNR0s.

        """
        return 2.0*np.log(rs) - 1.0/3.0*np.log(ps) + np.log(snr0s)

    @property
    def transit_selection_factor(self):
        """Probability of transit in a 1yr orbit about the sun.

        """
        return 0.001603891301 # Probability of detecting in yr orbit
                              # around sun.

    def ptransit(self, ps, ms, rs):
        """Returns the probability of a transit at the given periods around
        stars of the given masses and radii.

        """
        return self.transit_selection_factor*rs/(ms*np.square(ps))**(1.0/3.0)

    def pdetect(self, p, ps, rs, snr0s):
        """Returns the detection probability assuming that a planet transits
        given parameters ``p``, periods ``ps``, planetary radii
        ``rs``, and stellar SNRs ``snr0s``.

        """
        p = self.to_params(p)

        log_snrs = self.log_snr(ps, rs, snr0s)
        log_snrs = np.atleast_1d(log_snrs)

        pdets = np.zeros(log_snrs.shape)

        sel = (p['log_snr_min'] < log_snrs) & (log_snrs <= p['log_snr_max'])
        pdets[sel] = (log_snrs[sel]-p['log_snr_min'])/(p['log_snr_max'] - p['log_snr_min'])

        sel = (p['log_snr_max'] < log_snrs)
        pdets[sel] = 1.0

        return pdets

    def gaussian_selection_integral(self, mu, sigma, xmin, xmax):
        r"""Evaluates 

        .. math::

          \int_{x_\mathrm{min}}^{x_\mathrm{max}} dx\, \phi(x) \frac{x-x_\mathrm{min}}{x_\mathrm{max}-x_\mathrm{min}} + \int_{x_\mathrm{max}}^\infty dx \, \phi(x)

        """

        dx = xmax - xmin
        edenom = np.sqrt(2.0)*sigma

        mu_term = 0.5*mu*(ss.erf((mu-xmin)/edenom) - ss.erf((mu-xmax)/edenom))
        xterm = 0.5*(xmax*ss.erf((mu-xmax)/edenom) - xmin*ss.erf((mu-xmin)/edenom))
        sigma_term = 1.0/np.sqrt(2.0*np.pi)*sigma*(np.exp(-0.5*np.square((mu-xmin)/sigma)) - np.exp(-0.5*np.square((mu-xmax)/sigma)))

        return 1.0/dx*(mu_term + xterm + sigma_term) + 0.5

    def alpha(self, p, ms, rs, snr0s):
        """Returns the average probability over the planetary period-radius
        distribution of detecting a planet (transit probability times
        detection probability) about stars with the given masses,
        radii, and SNRs.

        """

        p = self.to_params(p)

        mu = p['mu']
        cm = self.covariance_matrix(p)
        gm = self.covariance_matrix(p, inv=True)

        gm_det = gm[0,0]*gm[1,1]-gm[0,1]*gm[0,1]

        mu_new = mu - np.array([2.0/3.0*gm[1,1]/gm_det, -2.0/3.0*gm[0,1]/gm_det])

        lognorm_norm = np.exp(-2.0/3.0*mu[0] + 2.0/9.0*gm[1,1]/gm_det)
        geom_factor = self.transit_selection_factor*rs/ms**(1.0/3.0)

        log_snr_mean = 2.0*mu_new[1] - 1.0/3.0*mu_new[0] + np.log(snr0s)
        log_snr_sigma = np.sqrt(4.0*cm[1,1] - 4.0/3.0*cm[0,1] + 1.0/9.0*cm[0,0])

        return lognorm_norm*geom_factor*self.gaussian_selection_integral(log_snr_mean, log_snr_sigma, p['log_snr_min'], p['log_snr_max'])

    def __call__(self, p):
        """Returns the posterior evaluated at ``p``.

        """

        p = self.to_params(p)

        lower_left = self.lower_left.copy()
        lower_left[0] = np.log(p['Pmin'])

        dx, dy = self.upper_right - lower_left
        gx, gy = p['gamma']

        # Priors bounds
        if p['R'] <= 0 or p['Rb'] <= 0:
            return np.NINF
        if np.any(p['mu'] < np.min(self.pts, axis=0)) or \
           np.any(p['mu'] > np.max(self.pts, axis=0)):
            return np.NINF
        if np.any(p['sigma'] <= 0):
            return np.NINF
        if p['theta'] < 0 or p['theta'] > np.pi/2.0:
            return np.NINF
        if p['log_snr_max'] <= p['log_snr_min']:
            return np.NINF
        if 1.0 - 0.5*np.abs(dx*gx) - 0.5*np.abs(dy*gy) < 0:
            return np.NINF
        if np.log(p['Pmin']) > self.upper_right[0]:
            return np.NINF

        alphas = self.alpha(p, self.systems['Mass'], self.systems['Radius'], self.systems['SNR0'])

        if np.any(alphas > 1):
            return np.NINF

        Rtotal = p['R']*np.sum(alphas) + p['Rb']*self.systems.shape[0]
        rhos = p['R']*self.pdetect(p, self.candidates['Period'], self.candidates['Radius'], self.candidates['SNR0'])*self.ptransit(self.candidates['Period'], self.candidates['Stellar_Mass'], self.candidates['Stellar_Radius'])*self.foreground_density(p, self.candidates['Period'], self.candidates['Radius']) + p['Rb']*self.background_density(p, self.candidates['Period'], self.candidates['Radius'])

        ll = np.sum(np.log(rhos)) - Rtotal

        lp = -np.sum(np.log(p['sigma'])) - 0.5*np.square(p['log_snr_min'] - np.log(3.0)) - 0.5*np.square(p['log_snr_max'] - np.log(11.0)) - 0.5*(np.log(p['R']*np.sum(alphas)) + np.log(p['Rb']))

        return ll + lp

    def draw_background(self, p, N):
        """Draw ``N`` systems from the background distribution specified by
        parameters ``p``.

        """

        p = self.to_params(p)

        lower_left = self.lower_left.copy()
        lower_left[0] = np.log(p['Pmin'])

        dx, dy = self.upper_right - lower_left
        gx, gy = p['gamma']
        V = dx*dy

        pmax = 1.0/V*(1.0 + 0.5*np.abs(dx*gx) + 0.5*np.abs(dy*gy))

        pts = []
        while len(pts) < N:
            x,y,z = np.random.random(size=3)

            x = lower_left[0] + dx*x
            y = lower_left[1] + dy*y
            z = z*pmax

            x = np.exp(x)
            y = np.exp(y)

            if z < self.background_density(p, x, y):
                pts.append(np.array([x,y]))

        return np.array(pts)

    def draw(self, p0, ids, masses, radii, snr0s):
        """Draw planets and background "detections" around the stars with the
        given masses, radii, and SNRs as described by the model
        parameters ``p0``.

        """

        p0 = self.to_params(p0)

        fs = []
        bs = []

        mu = p0['mu']
        cm = self.covariance_matrix(p0)

        # Draw foreground events
        n = np.random.poisson(p0['R']*ids.shape[0])
        pts = np.random.multivariate_normal(mean=mu, cov=cm, size=n)
        inds = np.random.randint(ids.shape[0], size=n)
        PS = np.exp(pts[:,0])
        RS = np.exp(pts[:,1])
        psel = self.pdetect(p0, PS, RS, snr0s[inds])*self.ptransit(PS, masses[inds], radii[inds])
        sel = np.random.random(size=n) < psel

        fps = PS[sel]
        frs = RS[sel]
        fids = ids[inds][sel]
        fmasses = masses[inds][sel]
        fradii = radii[inds][sel]
        fsnr0s = snr0s[inds][sel]
        
        # Draw background events
        n = np.random.poisson(p0['Rb']*ids.shape[0])
        ps_rs = self.draw_background(p0, n)
        inds = np.random.randint(ids.shape[0], size=n)

        syss = []

        for fp, fr, fid, fm, frad, fs in zip(fps, frs, fids, fmasses, fradii, fsnr0s):
            syss.append((int(fid), fp, fr, frad, fm, fs))

        for bp, br, bi in zip(ps_rs[:,0], ps_rs[:,1], inds):
            syss.append((int(ids[bi]), bp, br, radii[bi], masses[bi], snr0s[bi]))

        candidates = np.array(syss, dtype=np.dtype([('Kepler_ID', np.int),
                                                    ('Period', np.float),
                                                    ('Radius', np.float),
                                                    ('Stellar_Radius', np.float),
                                                    ('Stellar_Mass', np.float),
                                                    ('SNR0', np.float)]))

        return candidates

    def gradient(self, p):
        """Returns a numerical approximation to the gradient of the posterior
        at the point ``p``.

        """

        g = []
        for i in range(self.nparams):
            def f(x):
                pc = p.copy()
                pc[i] = x
                return self.__call__(pc)
            g.append(sm.derivative(f, p[i], dx=1e-5))

        return np.array(g)
