import bz2
import glob
import numpy as np
import os.path as op
import scipy.integrate as si
import scipy.special as ss
import scipy.stats as st
import warnings

class TooManyPointsError(Exception):
    def __init__(self, *args, **kwargs):
        super(TooManyPointsError, self).__init__(*args, **kwargs)

def integrate(f, xmin, xmax, ymin, ymax, epsrel):
    """Integrate :math:`f(x,y)` over the rectangular domain
    :math:`(x_\mathrm{min}, y_\mathrm{min}) < (x,y) < (x_\mathrm{max},
    y_\mathrm{max})`, to an (estimated) relative accuracy of
    ``epsrel``.

    :param f: Function of two arguments.  Must be able to take an
      array of shape ``(N, 2)`` and return an array of shape ``(N,)``.

    :param xmin: Lower limit in first argument.

    :param xmax: Upper limit in first argument.

    :param ymin: Lower limit in second argument.

    :param ymax: Upper limit in second argument.

    :param epsrel: Relative error tolerance.
    """

    xs = np.linspace(xmin, xmax, 3)
    ys = np.linspace(ymin, ymax, 3)
    k = 1

    XS,YS = np.meshgrid(xs, ys)
    FS = f(np.column_stack((XS.flatten(), YS.flatten()))).reshape(XS.shape)

    I = si.romb(si.romb(FS, axis=0, dx=ys[1]-ys[0]), axis=0, dx=xs[1]-xs[0])

    while True:
        FS_old = FS
        I_old = I

        k = k+1
        N = (1<<k) + 1

        if k > 13:
            # More than 8193 points in each dimension
            raise TooManyPointsError('more than 2^{0:d} points in each dimension'.format(k))

        xs = np.linspace(xmin, xmax, N)
        ys = np.linspace(ymin, ymax, N)

        XS, YS = np.meshgrid(xs, ys)
        FS = f(np.column_stack((XS.flatten(), YS.flatten()))).reshape(XS.shape)

        I = si.romb(si.romb(FS, axis=0, dx=ys[1]-ys[0]), axis=0, dx=xs[1]-xs[0])

        err = 2.0*np.abs(I-I_old)/(np.abs(I) + np.abs(I_old))

        if not (err > epsrel):
            break

    return I

def load_kepler_data(file):
    with bz2.BZ2File(file, 'r') as inp:
        header = inp.readline().split('|')
        inp.readline()
        cols = (header.index('Period'), header.index('Planet Radius'), header.index('Kepler Disposition'))
        data = np.genfromtxt(inp, usecols=cols, delimiter='|', dtype=np.dtype([('P', np.float), ('R', np.float), ('disp', np.str, 100)]))

    data = data[(~np.isnan(data['R'])) & (~np.isnan(data['P'])) & (data['disp'] == 'CANDIDATE')]

    return data

class Posterior(object):
    def __init__(self, p, r):
        self.p = p
        self.r = r
        self.pts = np.log(np.column_stack((p, r)))

    @property
    def dtype(self):
        return np.dtype([('A', np.float),
                         ('mu', np.float, 2),
                         ('sigma', np.float, 2),
                         ('theta', np.float),
                         ('mu_snr', np.float),
                         ('sigma_snr', np.float),
                         ('lower_left', np.float, 2),
                         ('upper_right', np.float, 2),
                         ('gamma', np.float, 2)])

    @property
    def pnames(self):
        return [r'A',
                r'$\mu_P$', r'$\mu_R$',
                r'$\sigma_1$', r'$\sigma_2$',
                r'$\theta$',
                r'$\mu_\rho$', r'$\sigma_\rho$',
                r'$P_\mathrm{min}$', r'$R_\mathrm{min}$',
                r'$P_\mathrm{max}$', r'$R_\mathrm{max}$', r'$\gamma_P$',
                r'$\gamma_R$']

    @property
    def nparams(self):
        return 14

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

    def geometric_pselect(self, pts):
        return 0.08256960301*np.exp(-2.0/3.0*pts[:,0])

    def snr_pselect(self, p, pts):
        p = self.to_params(p)
        pts = np.atleast_2d(pts)
        
        log_snr = 2.0*pts[:,1] - 1.0/3.0*pts[:,0]

        return 1.0/(1.0 + np.exp(-(log_snr - p['mu_snr'])/p['sigma_snr']))

    def pselect(self, p, pts):
        return self.geometric_pselect(pts)*self.snr_pselect(p, pts)

    def foreground_density(self, p, pts):
        p = self.to_params(p)

        cm = self.covariance_matrix(p, inv=True)

        xs = pts - p['mu']

        return 1.0/(2.0*np.pi*np.prod(p['sigma']))*np.exp(-0.5*np.sum(xs*np.dot(cm, xs.T).T, axis=1))

    def background_density(self, p, pts=None):
        p = self.to_params(p)
        if pts is None:
            pts = self.pts
            
        center = 0.5*(p['lower_left'] + p['upper_right'])
        dx = p['upper_right'] - p['lower_left']
        V = np.prod(dx)
        
        rhos = 1.0/V*(1.0 + np.dot(p['gamma'], (pts - center).T))

        sel = (pts < p['lower_left']) | (pts > p['upper_right'])
        sel = sel[:,0] | sel[:,1]

        rhos[sel] = 0.0
        return rhos

    def selected_density(self, p, pts):
        return self.pselect(p, pts)*self.foreground_density(p, pts)/self.alpha(p)

    def alpha(self, p):
        p = self.to_params(p)

        cm = self.covariance_matrix(p)

        xmin = p['mu'][0] - 10.0*cm[0,0]
        xmax = p['mu'][0] + 10.0*cm[0,0]
        ymin = p['mu'][1] - 10.0*cm[1,1]
        ymax = p['mu'][1] + 10.0*cm[1,1]

        xmin = min(xmin, np.min(self.pts[:,0]))
        xmax = max(xmax, np.max(self.pts[:,0]))
        ymin = min(ymin, np.min(self.pts[:,1]))
        ymax = max(ymax, np.max(self.pts[:,1]))

        return integrate(lambda pts: self.pselect(p, pts)*self.foreground_density(p, pts),
                         xmin, xmax, ymin, ymax, 1e-10)

    def __call__(self, p):
        p = self.to_params(p)

        dx, dy = p['upper_right'] - p['lower_left']
        gx, gy = p['gamma']

        # Priors bounds
        if p['A'] < 0 or p['A'] > 1:
            return np.NINF
        if np.any(p['mu'] < np.min(self.pts, axis=0)) or \
           np.any(p['mu'] > np.max(self.pts, axis=0)):
            return np.NINF
        if np.any(p['sigma'] <= 0):
            return np.NINF
        if p['theta'] < 0 or p['theta'] > np.pi/2.0:
            return np.NINF
        if p['sigma_snr'] <= 0:
            return np.NINF
        if np.any(p['upper_right'] <= p['lower_left']):
            return np.NINF
        if 1.0 - 0.5*np.abs(dx*gx) - 0.5*np.abs(dy*gy) < 0:
            return np.NINF

        alpha = self.alpha(p)

        pselects = self.pselect(p, self.pts)
        rho_fore = self.foreground_density(p, self.pts)
        rho_back = self.background_density(p, self.pts)

        rho = p['A']*pselects*rho_fore/alpha + (1-p['A'])*rho_back

        ll = np.sum(np.log(rho))

        lp = -np.sum(np.log(p['sigma'])) - np.log(p['sigma_snr']) + 0.5*np.log(np.sum(np.square(p['gamma'])))

        return ll + lp

    def draw(self, p, N):
        p = self.to_params(p)

        cm = self.covariance_matrix(p)

        nfore = np.random.binomial(N, p['A'])
        nback = N-nfore

        dx, dy = p['upper_right'] - p['lower_left']
        gx, gy = p['gamma']

        zmax = 1.0 + 0.5*np.abs(gx*dx) + 0.5*np.abs(gy*dy)
        d = np.array([dx, dy])

        back_pts = p['lower_left'] + d*np.random.random(size=(nback, 2))
        back_sel = zmax*np.random.random(size=nback) > self.background_density(p, back_pts)
        while np.any(back_sel):
            nnz = np.count_nonzero(back_sel)
            back_pts[back_sel, :] = p['lower_left'] + d*np.random.random(size=(nnz,2))
            back_sel[back_sel] = zmax*np.random.random(size=nnz) > self.background_density(p, back_pts[back_sel,:])

        fore_pts = np.random.multivariate_normal(mean=p['mu'], cov=cm, size=nfore)
        fore_sel = np.random.random(size=nfore) > self.pselect(p, fore_pts)

        while np.any(fore_sel):
            nnz = np.count_nonzero(fore_sel)
            fore_pts[fore_sel, :] = np.random.multivariate_normal(mean=p['mu'], cov=cm, size=nnz)
            fore_sel[fore_sel] = np.random.random(size=nnz) > self.pselect(p, fore_pts[fore_sel, :])

        pts = np.concatenate((fore_pts, back_pts), axis=0)

        return np.exp(pts[:,0]), np.exp(pts[:,1])
