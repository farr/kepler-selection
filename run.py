#!/usr/bin/env python

import argparse
import bz2
import emcee
import numpy as np
import os
import os.path as op
import plotutils.autocorr as ac
import posterior
import uuid

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--candidates', required=True, help='candidate file')
    parser.add_argument('--systems', required=True, help='system file')

    parser.add_argument('--outdir', default='.', help='output directory')

    parser.add_argument('--threads', type=int, default=15, help='number of threads')
    parser.add_argument('--walkers', type=int, default=100, help='number of walkers')

    parser.add_argument('--nstep', type=int, default=100, help='steps between saves')
    parser.add_argument('--nthin', type=int, default=10, help='steps between stored samples')

    parser.add_argument('--fburnin', type=float, default=0.1, help='fraction of samples to discard as burnin')

    args = parser.parse_args()

    candidates = np.genfromtxt(args.candidates,
                               names=True,
                               dtype=[np.int, np.float, np.float,
                                      np.float, np.float, np.float])
    systems = np.genfromtxt(args.systems,
                            names=True,
                            dtype=[np.int, np.float, np.float, np.float])

    logpost = posterior.Posterior(candidates, systems)

    sampler = emcee.EnsembleSampler(args.walkers, logpost.nparams,
                                    logpost, threads=args.threads)

    with bz2.BZ2File(op.join(args.outdir, 'chain.npy.bz2'), 'r') as inp:
        chain = np.load(inp)
    with bz2.BZ2File(op.join(args.outdir, 'lnprob.npy.bz2'), 'r') as inp:
        lnprob = np.load(inp)

    sampler._chain = chain
    sampler._lnprob = lnprob

    while True:
        sampler.run_mcmc(sampler.chain[:,-1,:], args.nstep, thin=args.nthin)

        print 'Saving state after ', sampler.chain.shape[1], ' samples....'
        print 'Autocorrelations are ', ac.emcee_chain_autocorrelation_lengths(sampler.chain, fburnin=args.fburnin)


        chain_name = uuid.uuid4().hex
        lnprob_name = uuid.uuid4().hex

        chain_name = chain_name + '.npy.bz2'
        lnprob_name = lnprob_name + '.npy.bz2'

        chain_file = op.join(args.outdir, chain_name)
        lnprob_file = op.join(args.outdir, lnprob_name)

        with bz2.BZ2File(chain_file, 'w') as out:
            np.save(out, sampler.chain)
        with bz2.BZ2File(lnprob_file, 'w') as out:
            np.save(out, sampler.lnprobability)
        
        os.rename(chain_file, op.join(args.outdir, 'chain.npy.bz2'))
        os.rename(lnprob_file, op.join(args.outdir, 'lnprob.npy.bz2'))
