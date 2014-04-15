import bz2
import glob
import numpy as np
import os.path as op
import scipy.integrate as si
import scipy.special as ss
import scipy.stats as st

class Posterior(object):
    def __init__(self, candidates, systems):
        self.candidates = candidates
        self.systems = systems
        self.pts = np.log(np.column_stack((candidates['Period'], candidates['Radius'])))

    @property
    def dtype(self):
        return np.dtype([('R', np.float),
                         ('Rb', np.float),
                         ('mu', np.float, 2),
                         ('sigma', np.float, 2),
                         ('theta', np.float),
                         ('log_snr_min', np.float),
                         ('log_snr_max', np.float),
                         ('lower_left', np.float, 2),
                         ('upper_right', np.float, 2),
                         ('gamma', np.float, 2)])

    @property
    def pnames(self):
        return [r'$R$', r'$R_b$',
                r'$\mu_P$', r'$\mu_R$',
                r'$\sigma_1$', r'$\sigma_2$',
                r'$\theta$',
                r'$\log\left(\rho_\mathrm{min}\right)$',
                r'$\log\left(\rho_\mathrm{max}\right)$',
                r'$P_\mathrm{min}$', r'$R_\mathrm{min}$',
                r'$P_\mathrm{max}$', r'$R_\mathrm{max}$', r'$\gamma_P$',
                r'$\gamma_R$']

    @property
    def nparams(self):
        return 15

    def to_params(self, p):
        return p.view(self.dtype).squeeze()

    def covariance_matrix(self, p, inv=False):
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
        p = self.to_params(p)

        pts = np.log(np.column_stack((ps, rs)))

        cm = self.covariance_matrix(p, inv=True)

        xs = pts - p['mu']

        return 1.0/(2.0*np.pi*np.prod(p['sigma']))*np.exp(-0.5*np.sum(xs*np.dot(cm, xs.T).T, axis=1))

    def background_density(self, p, ps, rs):
        p = self.to_params(p)

        ps = np.atleast_1d(ps)
        rs = np.atleast_1d(rs)

        pts = np.log(np.column_stack((ps, rs)))
            
        center = 0.5*(p['lower_left'] + p['upper_right'])
        dx = p['upper_right'] - p['lower_left']
        V = np.prod(dx)
        
        rhos = 1.0/V*(1.0 + np.dot(p['gamma'], (pts - center).T))

        sel = (pts < p['lower_left']) | (pts > p['upper_right'])
        sel = sel[:,0] | sel[:,1]

        rhos[sel] = 0.0
        return rhos

    def log_snr(self, ps, rs, snr0s):
        return 2.0*np.log(rs) - 1.0/3.0*np.log(ps) + np.log(snr0s)

    @property
    def transit_selection_factor(self):
        return 0.001603891301 # Probability of detecting in yr orbit
                              # around sun.

    def ptransit(self, ps, ms, rs):
        return self.transit_selection_factor*rs/(ms*np.square(ps))**(1.0/3.0)

    def pdetect(self, p, ps, rs, snr0s):
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
        dx = xmax - xmin
        edenom = np.sqrt(2.0)*sigma

        mu_term = 0.5*mu*(ss.erf((mu-xmin)/edenom) - ss.erf((mu-xmax)/edenom))
        xterm = 0.5*(xmax*ss.erf((mu-xmax)/edenom) - xmin*ss.erf((mu-xmin)/edenom))
        sigma_term = 1.0/np.sqrt(2.0*np.pi)*sigma*(np.exp(-0.5*np.square((mu-xmin)/sigma)) - np.exp(-0.5*np.square((mu-xmax)/sigma)))

        return 1.0/dx*(mu_term + xterm + sigma_term) + 0.5

    def alpha(self, p, ms, rs, snr0s):
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
        p = self.to_params(p)

        dx, dy = p['upper_right'] - p['lower_left']
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
        if np.any(p['upper_right'] <= p['lower_left']):
            return np.NINF
        if 1.0 - 0.5*np.abs(dx*gx) - 0.5*np.abs(dy*gy) < 0:
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
        p = self.to_params(p)

        dx, dy = p['upper_right'] - p['lower_left']
        gx, gy = p['gamma']
        V = dx*dy

        pmax = 1.0/V*(1.0 + 0.5*np.abs(dx*gx) + 0.5*np.abs(dy*gy))

        pts = []
        while len(pts) < N:
            x,y,z = np.random.random(size=3)

            x = p['lower_left'][0] + dx*x
            y = p['lower_left'][1] + dy*y
            z = z*pmax

            x = np.exp(x)
            y = np.exp(y)

            if z < self.background_density(p, x, y):
                pts.append(np.array([x,y]))

        return np.array(pts)

    def draw(self, p0, ids, masses, radii, snr0s):
        p0 = self.to_params(p0)

        fs = []
        bs = []

        mu = p0['mu']
        cm = self.covariance_matrix(p0)

        # Draw foreground events
        for id, m, r, s in zip(ids, masses, radii, snr0s):
            n = np.random.poisson(p0['R'])
            if n > 0:
                pts = np.random.multivariate_normal(mean=mu, cov=cm, size=n)

                for p in pts:
                    P = np.exp(p[0])
                    R = np.exp(p[1])

                    pselect = self.pdetect(p0, P, R, s)*self.ptransit(P, m, r)

                    if np.random.random() < pselect:
                        fs.append((int(id), P, R, r, m, s))

            n = np.random.poisson(p0['Rb'])
            if n > 0:
                ps_rs = self.draw_background(p0, n)

                for P, R in ps_rs:
                    bs.append((int(id), P, R, r, m, s))

        candidates = fs + bs
                    
        candidates = np.array(candidates, dtype=np.dtype([('Kepler_ID', np.int),
                                                          ('Period', np.float),
                                                          ('Radius', np.float),
                                                          ('Stellar_Radius', np.float),
                                                          ('Stellar_Mass', np.float),
                                                          ('SNR0', np.float)]))

        return candidates
                                                                     
        
