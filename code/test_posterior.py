import numpy as np
import scipy.integrate as si
import scipy.special as sp
import scipy.stats as ss

class TestPosterior(object):
    def __init__(self, xs):
        self._xs = xs

    @property
    def xs(self):
        return self._xs

    @property
    def dtype(self):
        return np.dtype([('mu', np.float),
                         ('sigma', np.float),
                         ('mu_det', np.float),
                         ('sigma_det', np.float)])

    @property
    def nparams(self):
        return 4

    @property
    def pnames(self):
        return [r'$\mu$', r'$\sigma$',
                r'$\mu_\mathrm{det}$',
                r'$\sigma_\mathrm{det}$']

    def to_params(self, p):
        p = np.atleast_1d(p)
        return p.view(self.dtype).squeeze()

    def _sigmoid(self, xs):
        return 1.0/(1.0 + np.exp(-xs))

    def pselect(self, p, xs):
        p = self.to_params(p)
        xs = np.atleast_1d(xs)

        return self._sigmoid((xs-p['mu_det'])/p['sigma_det'])

    def density(self, p, xs):
        p = self.to_params(p)

        return ss.norm.pdf(xs, loc=p['mu'], scale=p['sigma'])

    def alpha(self, p):
        p = self.to_params(p)

        xmin = min(np.min(self.xs), p['mu']-10.0*p['sigma'], p['mu_det']-10.0*p['sigma_det'])
        xmax = max(np.max(self.xs), p['mu']+10.0*p['sigma'], p['mu_det']+10.0*p['sigma_det'])

        return si.romberg(lambda x: self.pselect(p, x)*self.density(p, x),
                          xmin, xmax, divmax=100, vec_func=True)

    def __call__(self, p):
        p = self.to_params(p)

        if p['sigma'] <= 0:
            return np.NINF
        if p['sigma_det'] <= 0:
            return np.NINF

        alpha = self.alpha(p)

        probs = self.pselect(p, self.xs)*self.density(p, self.xs)

        return np.sum(np.log(probs)) - self.xs.shape[0]*np.log(alpha) - np.log(p['sigma']) - np.log(p['sigma_det'])

    def draw(self, p, N):
        p = self.to_params(p)

        xs = np.random.normal(loc=p['mu'], scale=p['sigma'], size=N)
        psel = self.pselect(p, xs)

        return xs[np.random.random(size=N) < psel]

    def selected_density(self, p, xs):
        alpha = self.alpha(p)

        return self.density(p, xs)*self.pselect(p, xs)/alpha
