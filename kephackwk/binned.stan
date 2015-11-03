functions {
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
}

data {
  int nobs;
  int nps;
  int nrs;

  real pbins[nps+1];
  real rbins[nrs+1];

  real pobs[nobs];
  real robs[nobs];
  real drobs[nobs];

  real alphas[nps, nrs];

  real rmin;
  real rmax;
}

transformed data {
  real log_rweights[nobs, nrs];
  real log_ca_rweights[nobs];
  int pinds[nobs];

  real dlogps[nps];
  real dlogrs[nrs];

  real dlogr_ca;
  real dlogp_ca;
  
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
}

parameters {
  real rawrate[nps, nrs];
  real<lower=0> dndlnpdlnr_ca;
  real mu;
  real<lower=0> sigma;
}

transformed parameters {
  real dndlnpdlnr[nps, nrs];

  /* rates ~ lognormal(mu, sigma) when rawrates ~ N(0,1) */
  for (i in 1:nps) {
    for (j in 1:nrs) {
      dndlnpdlnr[i,j] <- exp(rawrate[i,j]*sigma + mu);
    }
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

  for (i in 1:nps) {
    for (j in 1:nrs) {
      rawrate[i,j] ~ normal(mu, sigma);
    }
  }

  /* dndlnpdlnr_ca ~ 1/sqrt(...) */
  increment_log_prob(-0.5*log(dndlnpdlnr_ca));

  mu ~ normal(0.0, 10.0);
  sigma ~ cauchy(0.0, 10.0);
}
