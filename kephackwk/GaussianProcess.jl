module GaussianProcess

"""Returns a covariance matrix for the squared exponential kernel,
using a metric to compute the distance between the points.

`pts` an `(ndim, npts)` array of the points at which the kernel should
    be evaluated.

`var` the variance of the kernel

`metric` an `(ndim, ndim)` matrix which must be positive definite and
    is used as a metric for computing the distance that goes into the
    kernel.  

`wn_amp` gives the relative amplitude of a white-noise term compared
        to the exponential kernel in the process variance.  This
        enhances the stability of the process.

"""
function sq_exp_ker(pts, var, metric, wn_amp)
    ndim, npts = size(pts)

    ker = zeros(npts, npts)

    for j in 1:npts
        for i in 1:npts
            for k in 1:ndim
                for l in 1:ndim
                    @inbounds ker[i,j] += (pts[k,i]-pts[k,j])*metric[k,l]*(pts[l,i]-pts[l,j])
                end
            end
        end
    end

    var = var/(one(wn_amp) + wn_amp)
    for j in 1:npts
        for i in 1:npts
            @inbounds ker[i,j] = var*exp(-0.5*ker[i,j])
        end
    end
    
    for i in 1:npts
        @inbounds ker[i,i] += var*wn_amp
    end

    return ker
end

"""Squared exponential kernel GP log-likelihood.

`xs` an array of size `(ndim, npts)` giving the locations of the
        values `ys`

`ys` an array of size `(npts,)` giving the samples from the GP.

`mu` the mean of the GP

`var` the variance of the GP

`metric` an `(ndim, ndim)` positive-definite matrix defining the
        metric used to compute distances between the points `xs` in
        the GP kernel.

`wn_amp` gives the relative amplitude of a white-noise term in the
        process variance.

"""
function sq_exp_log_like(xs, ys, mu, var, metric, wn_amp)
    ker = sq_exp_ker(xs, var, metric, wn_amp)

    ker_fact = cholfact!(ker, :U)
    ker_U = ker_fact[:U]
    
    npts = size(ys, 1)
    zs = zeros(npts)
    for i in 1:npts
        @inbounds zs[i] = ys[i] - mu
    end
    
    ld = 0.0
    for i in 1:npts
        @inbounds ld += log(ker_U[i,i])
    end
    
    return -0.5*npts*log(2.0*pi) - ld - 0.5*dot(zs, ker_fact\zs)
end

"""Return a draw from the GP with squared exponential kernel at the
given points.

`xs` array of size `(ndim, npts)` giving the locations at which the
        process is to be evaluated.

`mu` the mean of the process

`var` the variance of the process

`metric` a positive-definite matrix of size `(ndim, ndim)` giving the
        metric used to evaluate distances in the squared exponential
        kernel.

`wn_amp` gives the relative amplitude of a white-noise term in the
        process.

Returns an array of size `(npts,)` that is a draw from the GP.

"""
function sq_exp_rand(xs, mu, var, metric, wn_amp)
    ker = sq_exp_ker(xs, var, metric, wn_amp)

    npts = size(xs, 2)

    z = randn(npts)

    F = chol(ker, :L)

    return mu + F*z
end

end
