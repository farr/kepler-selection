/* In this model, we are fitting a histogrammed occurrence rate for
   planets to Kepler measurements of periods and radii of planets.  We
   assume that the detection efficiency integrated across each
   distribution-function bin is constant and known, that the periods
   of the planets are measured perfectly, and that the true radii are
   log-normally distributed about the measured radii with s.d. in the
   log equal to the quoted relative radius uncertainty.

   We regularise the otherwise-independent estimates of the
   distribution function in bins using a squared-exponential GP prior
   on the log of the planet density in bins.  The metric (in
   log-space), amplitude, and white-noise contribution to the GP
   covariance are fitted as hyperparameters.

   The likelihood at fixed P, R is inhomogeneous Poisson:

   L = prod_k(alpha[i_k, j_k]*dN/dlnPdlnR[i_k, j_k]) exp(-sum_{ij} alpha[i,j]*dN/dlnPdlnR[i,j]*dlogP[i]*dlogR[j])

   where i_k, j_k are the bins into which planet k's period and radius
   fall.  If we use a log-normal distribution for the true radius,
   this can be analytically integrated over planet radii.  The result
   is an array of factors for each planet and each radius bin
   corresponding to the probability mass for that planet in the
   corresponding bin:

   <L> = prod_k(sum_j(alpha[i,j]*dN/dlnPdlnR[i,j]*factor[k,j])) exp(...)

   This is the likelihood we use below.  

   We include one "catchall" bin that extends between rmin and rmax,
   which "catches" all the probability mass for the radii that does
   not land in one of the other histogram bins.

 */

functions {
  /* Returns the index at which one should insert x into the array ys
     to maintain the sorted order. */
  int index(real x, real[] ys) {
    int n;

    n <- size(ys);

    if (x < ys[1]) {
      return 0;
    } else if (x >= ys[n]) {
      return n+1;
    } else {
      int imin;
      int imax;

      imin <- 1;
      imax <- n;

      while (imax - imin > 1) {
	int imid;

	imid <- imin + (imax - imin)/2;

	if (x >= ys[imid]) {
	  imin <- imid;
	} else {
	  imax <- imid;
	}
      }

      return imin;
    }
  }

  /* Returns the squared-exponential covariance matrix with variance
     sigma*sigma, metric g, and a white-noise term whose amplitude is
     sigma*sigma*wn_frac.  */
  matrix gp_cov(vector[] pts, matrix g, real sigma, real wn_frac, int N) {
    matrix[N,N] cov;

    for (i in 1:N) {
      vector[2] pti;

      pti <- pts[i];

      for (j in 1:N) {
	vector[2] dp;
	real r2;

	dp <- pti - pts[j];

	r2 <- g[1,1]*dp[1]*dp[1] + 2.0*g[1,2]*dp[1]*dp[2] + g[2,2]*dp[2]*dp[2];

	cov[i,j] <- sigma*sigma*exp(-0.5*r2);
      }
    }

    for (i in 1:N) {
      cov[i,i] <- cov[i,i]*(1.0 + wn_frac);
    }

    return cov;
  }
}

data {
  int nobs; /* Number of candidates. */
  int nps; /* Number of period bins. */
  int nrs; /* Number of radius bins. */

  real pbins[nps+1]; /* Period bin boundaries */
  real rbins[nrs+1]; /* Radius bin boundaries */

  real pobs[nobs]; /* Observed periods */
  real robs[nobs]; /* Observed radii */
  real drobs[nobs]; /* Uncertainties on the radii */

  real alphas[nps, nrs]; /* Selection function integrated across each
			    bin. */

  real rmin; /* Minimum radius allowed in analysis. */
  real rmax; /* Maximum radius allowed in analysis. */
}

transformed data {
  real log_rweights[nobs, nrs]; /* The weights from integrating the
				   planet radius probability mass in
				   each bin.*/
  real log_ca_rweights[nobs]; /* Weights for each planet in the
				 "catchall" bin. */

  int pinds[nobs]; /* Bin index for each planet's period */

  real dlogps[nps]; /* Width (in log-space) of the period bins. */
  real dlogrs[nrs]; /* Width (in log-space) of the radius bins. */

  real dlogr_ca; /* Width of catchall bin in R, P. */
  real dlogp_ca;

  vector[2] log_binpts[nps*nrs]; /* 2D vectors of bin centres, ordered
				    in memory like dndlnpdlnr. */
  
  for (i in 1:nps) {
    dlogps[i] <- log(pbins[i+1]) - log(pbins[i]);
  }

  for (i in 1:nrs) {
    dlogrs[i] <- log(rbins[i+1]) - log(rbins[i]);
  }

  dlogr_ca <- log(rbins[1]) - log(rmin) + (log(rmax) - log(rbins[nrs+1]));
  dlogp_ca <- log(pbins[nps+1]) - log(pbins[1]);
  
  for (i in 1:nobs) {
    real mu;
    real sigma;
    real cdfs[nrs+1];
    real d;

    mu <- log(robs[i]);
    sigma <- drobs[i]/robs[i];

    for (j in 1:nrs+1) {
      cdfs[j] <- lognormal_cdf(rbins[j], mu, sigma);
    }

    for (j in 1:nrs) {
      d <- cdfs[j+1] - cdfs[j];

      if (d < 0.0) {
	log_rweights[i,j] <- negative_infinity();
      } else {
	log_rweights[i,j] <- log(d);
      }
    }

    d <- cdfs[1] + (1.0 - cdfs[nrs+1]);
    if (d < 0.0) {
      log_ca_rweights[i] <- negative_infinity();
    } else {
      log_ca_rweights[i] <- log(d);
    }
  }

  for (i in 1:nobs) {
    pinds[i] <- index(pobs[i], pbins);
  }

  for (i in 1:nps) {
    for (j in 1:nrs) {
      log_binpts[(i-1)*nrs + j][1] <- log(0.5*(pbins[i+1] + pbins[i]));
      log_binpts[(i-1)*nrs + j][2] <- log(0.5*(rbins[j+1] + rbins[j]));
    }
  }
}

parameters {
  real rawrate[nps, nrs]; /* Distributed as N(0,1).  */
  real<lower=0> dndlnpdlnr_ca; /* Catchall rate density */
  real mu; /* Constant mean of regularising GP. */
  real<lower=0> sigma; /* s.d. of regularising GP. */
  real<lower=0.01, upper=100.0> wn_frac; /* Fractional white noise
					    amplitude in GP. */
  cov_matrix[2] metric; /* Metric for GP. */
}

transformed parameters {
  real dndlnpdlnr[nps, nrs]; /* Rate density ~ exp(mu + L*rawrate) for
				cov = L*L^T (GP prior). */
  {
    vector[nps*nrs] y;
    vector[nps*nrs] z; 
    matrix[nps*nrs, nps*nrs] L;
    matrix[nps*nrs, nps*nrs] cov; 

    y <- to_vector(to_array_1d(rawrate));
    cov <- gp_cov(log_binpts, metric, sigma, wn_frac, nps*nrs);
    L <- cholesky_decompose(cov);
    z <- L*y;

    for (i in 1:nps)
      for (j in 1:nrs)
	dndlnpdlnr[i,j] <- exp(mu + z[(i-1)*nrs + j]);
  }
}

model {
  real excounts[nps, nrs];
  real ca_excounts;

  /* Accumulate expected counts. */
  for (i in 1:nps) {
    for (j in 1:nrs) {
      excounts[i,j] <- dndlnpdlnr[i,j]*alphas[i,j]*dlogps[i]*dlogrs[j];
    }
  }
  /* exp(-N) in Poisson: */
  ca_excounts <- dndlnpdlnr_ca*dlogp_ca*dlogr_ca;
  increment_log_prob(-sum(to_matrix(excounts)) - ca_excounts);

  /* Product over observations of marginal density. */
  for (i in 1:nobs) {
    int j;
    real log_densities[nrs + 1];

    j <- pinds[i];

    for (k in 1:nrs) {
      log_densities[k] <- log_rweights[i,k] + log(alphas[j,k]) + log(dndlnpdlnr[j,k]);
    }
    log_densities[nrs+1] <- log_ca_rweights[i] + log(dndlnpdlnr_ca);
    increment_log_prob(log_sum_exp(log_densities));
  }

  /* Implies rate follows GP. */
  to_array_1d(rawrate) ~ normal(0,1);

  /* dndlnpdlnr_ca ~ 1/sqrt(...) */
  increment_log_prob(-0.5*log(dndlnpdlnr_ca));

  /* Broad priors on mu and sigma. */
  mu ~ normal(0.0, 10.0);
  sigma ~ cauchy(0.0, 10.0);

  /* wn_frac ~ 1/wn_frac */
  increment_log_prob(-log(wn_frac));

  /* <g> = I under prior. */
  metric ~ wishart(2, (1.0/2.0)*diag_matrix(rep_vector(1.0,2)));
}
