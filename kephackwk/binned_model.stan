/* It turns out that the discreteness of the posterior (because the
   distribution function is assumed constant in P-R bins) causes the
   HMC sampler to fail. */

functions {
  int binsearch(real x, real[] xs) {
    int i;
    int n;

    n <- size(xs);

    if (x < xs[1]) {
      return 0;
    } else if (x >= xs[n]) {
      return n;
    } else {
      int ilow;
      int ihigh;

      ilow <- 1;
      ihigh <- n;

      while (ihigh - ilow > 1) {
	int imid;

	imid <- ilow + (ihigh - ilow)/2;

	if (x < xs[imid]) {
	  ihigh <- imid;
	} else {
	  ilow <- imid;
	}
      }

      return ilow; // Index of the bin x belongs to
    }
  }
}

data {
  int Ncand; // Number of candidates
  int Npbins; // Number of (uniform-in-log) bins in p
  int Nrbins; // Number of (uniform-in-log) bins in r

  real pbins[Npbins+1]; // Assumed to be sorted bin boundaries in P
  real rbins[Nrbins+1]; // Assumed to be sorted bin boundaries in R

  real pobs[Ncand]; // Observed periods
  real robs[Ncand]; // Observed radii
  real drobs[Ncand]; // Observed radial uncertainties

  real alphas[Npbins, Nrbins]; // Average detection efficiency over each bin
}

transformed data {
  real drrel[Ncand];
  int ips[Ncand];

  for (i in 1:Ncand) {
    drrel[i] <- drobs[i]/robs[i];
  }

  for (i in 1:Ncand) {
    ips[i] <- binsearch(pobs[i], pbins);
  }
}

parameters {
  vector[Ncand] logrs_raw;

  real<lower=0> catchall_rate;
  /* Raw value, distributed normal(0,1); transform using mu and
     sigma to dndlogpdlogr. */
  matrix[Npbins, Nrbins] logdndlogpdlogr_raw;

  /* Very basic multilevel prior; simple shrinkage. */
  real mu;
  real<lower=0.0> sigma;
}

transformed parameters {
  matrix[Npbins, Nrbins] dndlogpdlogr;
  vector[Ncand] rs;

  for (i in 1:Npbins)
    for (j in 1:Nrbins)
      dndlogpdlogr[i,j] <- exp(mu + sigma*logdndlogpdlogr_raw[i,j]);

  for (i in 1:Ncand)
    rs[i] <- exp(log(robs[i]) + drrel[i]*logrs_raw[i]);
}

model {
  int counts[Npbins, Nrbins];
  int total_counts;
  int noutside;
  matrix[Npbins, Nrbins] excounts;

  for (j in 1:Nrbins) {
    for (i in 1:Npbins) {
      real dlogp;
      real dlogr;

      dlogp <- log(pbins[i+1]) - log(pbins[i]);
      dlogr <- log(rbins[j+1]) - log(rbins[j]);
      
      excounts[i,j] <- alphas[i,j]*dndlogpdlogr[i,j]*dlogp*dlogr;
    }
  }
    
  /* The discontinuity in parameters will reduce sampling efficiency,
     but that can't be avoided for this model.  Hopefully it won't be
     too bad. */
  total_counts <- 0;
  for (k in 1:Ncand) {
    if (ips[k] >= 1 && ips[k] <= Npbins) {
      int j;
	
      j <- binsearch(rs[k], rbins);

      if (j >= 1 && j <= Nrbins) {
	counts[ips[k],j] <- counts[ips[k],j] + 1;
	total_counts <- total_counts + 1;
      }
    }
  }
  noutside <- Ncand - total_counts;

  /* Counts in each bin are Poisson. */
  for (i in 1:Npbins)
    for (j in 1:Nrbins)
      counts[i,j] ~ poisson(excounts[i,j]);
  noutside ~ poisson(catchall_rate);

  /* True rs are log-normal wrt observed rs */
  logrs_raw ~ normal(0.0, 1.0);

  /* Catchall rate has 1/sqrt(R) prior */
  increment_log_prob(-0.5*log(catchall_rate));

  /* Basic multilevel prior on the rates. */
  to_array_1d(logdndlogpdlogr_raw) ~ normal(0,1);

  /* Very broad prior on mu. */
  mu ~ normal(0.0, log(100.0));

  /* Half-cauchy prior on sigma. */
  sigma ~ cauchy(0.0, log(100.0));
}
