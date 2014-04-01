import numpy as np
import scipy.special as ss

def integrate_fixed(f, xmin, xmax, ymin, ymax, N):
    """Integrate the function ``f`` over a two-dimensional box with ``N``
    subdivisions along each axis.

    """
    xs = np.linspace(xmin, xmax, N+1)
    ys = np.linspace(ymin, ymax, N+1)

    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]

    XS, YS = np.meshgrid(xs, ys)
    pts = np.column_stack((XS.flatten(), YS.flatten()))

    fs = f(pts).reshape((N+1, N+1))

    return 0.25*dx*dy*np.sum(fs[:-1, :-1] + fs[:-1, 1:] + fs[1:, :-1] + fs[1:, 1:])

def integrate(f, xmin, xmax, ymin, ymax, epsrel):
    """Integrate ``f`` over a two-dimensional box with estimated relative
    error ``epsrel``.  ``f`` should take array arguments of the shape
    ``(Npts, 2)``.

    """
    N = 1
    flow = integrate_fixed(f, xmin, xmax, ymin, ymax, N)

    N = 2
    fhigh = integrate_fixed(f, xmin, xmax, ymin, ymax, N)

    fbest = 4.0/3.0*fhigh - 1.0/3.0*flow

    rel_err = 2.0*np.abs(fbest - fhigh)/(np.abs(fbest) + np.abs(fhigh))

    while rel_err >= epsrel:
        N *= 2

        flow = fhigh
        fbest_old = fbest
        
        fhigh = integrate_fixed(f, xmin, xmax, ymin, ymax, N)
        fbest = 4.0/3.0*fhigh - 1.0/3.0*flow

        rel_err = 2.0*np.abs(fbest-fbest_old)/(np.abs(fbest) + np.abs(fbest_old))

    return fbest

class Posterior(object):
    def __init__(self, pts):
        self._pts = pts

    @property
    def pts(self):
        return self._pts

    @property
    def dtype(self):
        return np.dtype([('A', np.float),
                         ('mu', np.float, 2),
                         ('sigma', np.float, 2),
                         ('theta', np.float),
                         ('p0', np.float),
                         ('mu_rho', np.float),
                         ('sigma_rho', np.float),
                         ('lower_left', np.float, 2),
                         ('upper_right', np.float, 2),
                         ('gamma', np.float, 2)])

    @property
    def pnames(self):
        return [r'$A$', r'$\mu_P$', r'$\mu_R$', r'$\sigma_1$', r'$\sigma_2$',
                r'$\theta$', r'$p_0$', r'$\mu_\rho$', r'$\sigma_\rho$',
                r'$\ln P_\mathrm{min}$', r'$\ln R_\mathrm{min}$',
                r'$\ln P_\mathrm{max}$', r'$\ln R_\mathrm{max}$',
                r'$\gamma_0$', r'$\gamma_1$']

    @property
    def nparams(self):
        return 15

    def to_params(self, p):
        return np.atleast_1d(p).view(self.dtype).squeeze()

    def covariance_matrix(self, p):
        p = self.to_params(p)

        d = np.diag(p['sigma']*p['sigma'])

        ct = np.cos(p['theta'])
        st = np.sin(p['theta'])

        r = np.array([[ct, -st], [st, ct]])

        return np.dot(r, np.dot(d, r.T))

    def inverse_covarance_matrix(self, p):
        p = self.to_params(p)

        d = np.diag(1.0 / (p['sigma']*p['sigma']))

        ct = np.cos(p['theta'])
        st = np.sin(p['theta'])

        r = np.array([[ct, -st], [st, ct]])

        return np.dot(r, np.dot(d, r.T))

    def geometric_probs(self, p, pts):
        p = self.to_params(p)

        return p['p0']/np.exp(2.0/3.0*pts[:,0])

    def snr_selection_probs(self, p, pts):
        p = self.to_params(p)
        log_snr = 2.0*pts[:,1] - 1.0/3.0*pts[:,0]

        return 0.5*(1.0 + ss.erf((log_snr - p['mu_rho'])/p['sigma_rho']))

    def pselect(self, p, pts):
        return self.geometric_probs(p, pts)*self.snr_selection_probs(p, pts)
        
    def background_density(self, p, pts):
        p = self.to_params(p)

        center = 0.5*(p['lower_left'] + p['upper_right'])
        dx = p['upper_right'] - p['lower_left']
        V = np.prod(dx)
        
        rhos = 1.0/V*(1.0 + np.dot(p['gamma'], (pts - center).T))

        sel = (pts < p['lower_left']) | (pts > p['upper_right'])
        sel = sel[:,0] | sel[:,1]

        rhos[sel] = 0.0
        return rhos

    def true_foreground_density(self, p, pts):
        p = self.to_params(p)

        cm = self.inverse_covarance_matrix(p)

        xs = pts - p['mu']

        return 1.0/(2.0*np.pi*np.prod(p['sigma']))*np.exp(-0.5*np.sum(xs*np.linalg.solve(cm, xs.T).T, axis=1))

    def unnormed_foreground_density(self, p, pts):
        return self.pselect(p, pts)*self.true_foreground_density(p, pts)

    def foreground_density_norm(self, p):
        p = self.to_params(p)

        cm = self.covariance_matrix(p)

        xmin = min(p['mu'][0] - 5.0*cm[0,0], np.min(self.pts[:,0]))
        xmax = max(p['mu'][0] + 5.0*cm[0,0], np.max(self.pts[:,0]))
        ymin = min(p['mu'][1] - 5.0*cm[1,1], np.min(self.pts[:,1]))
        ymax = max(p['mu'][1] + 5.0*cm[1,1], np.max(self.pts[:,1]))

        return 1.0/integrate((lambda pts: self.unnormed_foreground_density(p, pts)), xmin, xmax, ymin, ymax, 1e-8)

    def log_prior(self, p):
        p = self.to_params(p)

        if p['A'] < 0 or p['A'] > 1:
            return np.NINF
        if np.any(p['sigma'] < 0):
            return np.NINF
        if p['theta'] < 0 or p['theta'] > np.pi/2.0:
            return np.NINF
        if p['p0'] < 0 or np.any(self.geometric_probs(p, self.pts) > 1):
            return np.NINF
        if p['sigma_rho'] < 0:
            return np.NINF
        if np.any(p['lower_left'] >= p['upper_right']):
            return np.NINF

        pts = np.array([p['lower_left'], p['upper_right'],
                        [p['lower_left'][0], p['upper_right'][1]],
                        [p['upper_right'][0], p['lower_left'][1]]])

        if np.any(self.background_density(p, pts) < 0):
            return np.NINF

        # Restrict mean to be within distribution
        if np.any(p['mu'] < np.min(self.pts, axis=0)) or \
           np.any(p['mu'] > np.max(self.pts, axis=0)):
            return np.NINF

        lp = 0.0

        lp -= 0.5*(np.log(p['A']) + np.log1p(-p['A']))
        lp -= np.sum(np.log(p['sigma']))
        lp -= 2.0*np.log(p['sigma_rho'])

        return lp

    def log_likelihood(self, p):
        p = self.to_params(p)

        alpha = self.foreground_density_norm(p)

        rho_fore = alpha*self.unnormed_foreground_density(p, self.pts)
        rho_back = self.background_density(p, self.pts)

        rho = p['A']*rho_fore + (1-p['A'])*rho_back

        return np.sum(np.log(rho))

    def __call__(self, p):
        lp = self.log_prior(p)

        if lp == np.NINF:
            return np.NINF
        else:
            return self.log_likelihood(p) + lp
        
    def pfore(self, p, pts):
        alpha = self.foreground_density_norm(p)

        rho_fore = alpha*self.unnormed_foreground_density(p, pts)
        rho_back = self.background_density(p, pts)

        return rho_fore / (rho_fore + rho_back)

    def draw_candidates(self, p, N=None):
        p = self.to_params(p)

        cm = self.covariance_matrix(p)

        if N is None:
            N = self.pts.shape[0]

        Npts = np.random.poisson(N)

        pts = []
        while len(pts) < Npts:
            if np.random.random() < p['A']:
                # Drawing a planet
                while True:
                    logPR = np.random.multivariate_normal(mean=p['mu'], cov=cm)
                    pselect = self.pselect(p, np.array([logPR]))

                    if np.random.random() < pselect:
                        pts.append(logPR)
                        break
            else:
                # Drawing a background
                x0, y0 = p['lower_left']
                dx, dy = p['upper_right'] - p['lower_left']
                gx, gy = p['gamma']

                xmid = x0+dx/2.0
                ymid = y0+dy/2.0

                rho_max = 1.0/(dx*dy)*(1.0 + 0.5*np.abs(gx*dx) + 0.5*np.abs(gy*dy))

                while True:
                    x = x0 + dx*np.random.random()
                    y = x0 + dy*np.random.random()
                    z = rho_max*np.random.random()

                    pt = np.array([x,y])

                    rho = self.background_density(p, np.array([pt]))

                    if rho > z:
                        pts.append(pt)
                        break

        return np.array(pts)
