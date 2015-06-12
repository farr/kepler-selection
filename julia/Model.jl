module Model

using Ensemble

function log_odds(snr, snr_half, snr_width)
    0.5/(snr_half*snr_width)*(snr*snr - snr_half*snr_half)
end

function log_pselect(snr, snr_half, snr_width)
    lo = log_odds(snr, snr_half, snr_width)

    lo - Stats.logsumexp(lo, 0.0)
end

function log_pnselect(snr, snr_half, snr_width)
    lo = log_odds(snr, snr_half, snr_width)

    -Stats.logsumexp(lo, 0.0)
end

function transit_snr(P, R, snr0)
    
end

end
