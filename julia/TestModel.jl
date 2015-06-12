module TestModel

using Ensemble

draw_noise(mu, sigma, n) = exp(mu + sigma*randn(n))
draw_signal(mu, sigma, n) = draw_noise(mu, sigma, n)

log_odds(snr, snr_half, snr_width) = 0.5/(snr_half*snr_width)*(snr*snr - snr_half*snr_half)

function log_pselect(x, n, snr_half, snr_width)
    snr = x/n

    lo = log_odds(snr, snr_half, snr_width)

    lo - Stats.logsumexp(lo, 0.0)
end

function log_pnselect(x, n, snr_half, snr_width)
    snr = x/n

    lo = log_odds(snr, snr_half, snr_width)

    -Stats.logsumexp(lo, 0.0)
end

function draw(lambda, mu, sigma, snr_half, snr_width, ns)
    all_xs = map(ns) do n
        xs = draw_signal(mu, sigma, Stats.randpoi(lambda))
        psel = Float64[exp(log_pselect(x, n, snr_half, snr_width)) for x in xs]
        rs = rand(size(psel))
        xs_det = xs[rs .< psel]
        xs_ndet = xs[rs .>= psel]
        (xs_det, xs_ndet)
    end

    xs_det = [x[1] for x in all_xs]
    xs_ndet = [x[2] for x in all_xs]

    (xs_det, xs_ndet)
end

function lognormal_logpdf(x, mu, sigma)
    lx = log(x)
    y = lx-mu

    -0.91893853320467274178 - log(sigma) - lx - 0.5*y*y/(sigma*sigma)
end

const lambda0 = 10
const mu0 = log(7.0)
const sigma0 = 2.0/7.0

const mu_noise = log(1.0)
const sigma_noise = 0.2

const snr_half0 = 7.0
const snr_width0 = 1.0

const p0 = Float64[log(lambda0), mu0, log(sigma0), log(snr_half0), log(snr_width0)]

function make_lnprob(xsdet, ns)
    function lnprob(params, xsndet)
        log_lambda = params[1]
        mu = params[2]
        log_sigma = params[3]
        log_snr_half = params[4]
        log_snr_width = params[5]

        lambda = exp(log_lambda)
        sigma = exp(log_sigma)
        snr_half = exp(log_snr_half)
        snr_width = exp(log_snr_width)

        ll = 0.0

        for i in eachindex(xsdet)
            xs = xsdet[i]
            n = ns[i]

            for x in xs
                ll = ll + log_lambda + log_pselect(x, n, snr_half, snr_width) + lognormal_logpdf(x, mu, sigma)
            end
        end

        for i in eachindex(xsndet)
            xsn = xsndet[i]
            n = ns[i]

            for x in xsn
                ll = ll + log_lambda + log_pnselect(x, n, snr_half, snr_width) + lognormal_logpdf(x, mu, sigma)
            end
        end

        ll = ll - lambda*length(ns)

        lp = 0.0
        lp = lp - 0.5*log_lambda

        dsh = log_snr_half - log(snr_half0)
        lp = lp - 0.5*dsh*dsh

        dsw = log_snr_width - log(snr_width0)
        lp = lp - 0.5*dsw*dsw

        ll + lp
    end
end

function make_gibbsupdate(ns)
    function gibbsupdate(params, gs)
        log_lambda = params[1]
        mu = params[2]
        log_sigma = params[3]
        log_snr_half = params[4]
        log_snr_width = params[5]

        lambda = exp(log_lambda)
        sigma = exp(log_sigma)
        snr_half = exp(log_snr_half)
        snr_width = exp(log_snr_width)

        map(ns) do n
            xs = draw_signal(mu, sigma, Stats.randpoi(lambda))
            pnsel = [exp(log_pnselect(x, n, snr_half, snr_width)) for x in xs]
            rs = rand(size(pnsel, 1))
            xs[rs .< pnsel]
        end
    end
end

end
