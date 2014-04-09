#!/usr/bin/env python

import bz2
import emcee
import numpy as np
import os
import os.path as op
import plotutils.autocorr as ac
import posterior
import uuid

if __name__ == '__main__':
    threads = 3
    walkers = 100
    nstep = 100
    nthin = 10

    data = np.loadtxt('../test/ps-rs.dat.bz2')

    logpost = posterior.Posterior(data[:,0], data[:,1])

    sampler = emcee.EnsembleSampler(walkers, logpost.nparams, logpost, threads=threads)

    with bz2.BZ2File(op.join('..', 'test', 'chain.npy.bz2'), 'r') as inp:
        chain = np.load(inp)
    with bz2.BZ2File(op.join('..', 'test', 'lnprob.npy.bz2'), 'r') as inp:
        lnprob = np.load(inp)

    sampler._chain = chain
    sampler._lnprob = lnprob

    while True:
        sampler.run_mcmc(sampler.chain[:,-1,:], nstep, thin=nthin)

        istart = int(round(0.1*sampler.chain.shape[1]))

        print 'Saving state after ', sampler.chain.shape[1], ' samples....'
        print 'Autocorrelations are ', ac.emcee_chain_autocorrelation_lengths(sampler.chain[:,istart:,:])


        chain_name = uuid.uuid4().hex
        lnprob_name = uuid.uuid4().hex

        chain_name = chain_name + '.npy.bz2'
        lnprob_name = lnprob_name + '.npy.bz2'

        chain_file = op.join('..', 'test', chain_name)
        lnprob_file = op.join('..', 'test', lnprob_name)

        with bz2.BZ2File(chain_file, 'w') as out:
            np.save(out, sampler.chain)
        with bz2.BZ2File(lnprob_file, 'w') as out:
            np.save(out, sampler.lnprobability)
        
        os.rename(chain_file, op.join('..', 'test', 'chain.npy.bz2'))
        os.rename(lnprob_file, op.join('..', 'test', 'lnprob.npy.bz2'))
