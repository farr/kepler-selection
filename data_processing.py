import bz2
import glob
import numpy as np
import os.path as op

def load_kic_table(file):
    """Returns a numpy array with the fields ``kic_kepler_id``,
    ``kic_logg``, and ``kic_radius`` from the given file object.

    """
    header = file.readline().split('|')

    colnames = ['kic_kepler_id', 'kic_logg', 'kic_radius']

    cols = tuple([header.index(s) for s in colnames])

    data = np.genfromtxt(file, delimiter='|', usecols=cols,
                         dtype=np.dtype([(n, np.float) for n in colnames]))

    return data

def load_quarter(file):
    """Returns a numpy array with the fields 

      * ``ID``
      * ``cdpp12``
      * ``program``
      * ``logg``
      * ``rad``

    from reading the given file object, which should point to a
    "quarter" file from MAST.  The ``cdpp12`` field is the 12-hour
    CDPP, in ppm; the ``logg`` field is
    :math:`\log_{10}\frac{g}{\mathrm{cm}/\mathrm{s}^2}`; the ``rad``
    field is the radius in solar radii.

    """

    file.readline()
    header = file.readline().split()

    colnames = ['ID', 'cdpp12', 'program', 'logg', 'rad']
    coltypes = [(np.int,), (np.float,), (np.str, 20), (np.float, ), (np.float, )]

    dtype = np.dtype([(n,) + t for n, t in zip(colnames, coltypes)])

    cols = tuple([header.index(s) for s in colnames])

    data = np.genfromtxt(file, usecols=cols, dtype=dtype)

    return data

def load_candidates(file):
    """Returns a numpy array with the fields 

      * ``kepid``
      * ``koi_disposition``
      * ``koi_period``
      * ``koi_ror``

    read from the given file object, which should point to a
    tab-delimited candidate file from MAST.  The ``koi_period`` field
    is the period in days.

    """

    header = file.readline().split('\t')

    colnames = ['kepid', 'koi_disposition', 'koi_period', 'koi_ror']
    coltypes = [(np.int,), (np.str, 100), (np.float,), (np.float,)]

    dtype = np.dtype([(n,) + t for n,t in zip(colnames, coltypes)])

    cols = tuple([header.index(s) for s in colnames])

    data = np.genfromtxt(file, usecols=cols, dtype=dtype, delimiter='\t')

    return data

def filter_candidates(cands):
    """Takes a candidate array (for example, read by
    :func:`load_candidates`), and filters out any objects that are not
    candidates or confirmed planets, or bad radius or period
    measurements.

    """

    sel = ((cands['koi_disposition'] == 'CANDIDATE') | (cands['koi_disposition'] == 'CONFIRMED')) & (~np.isnan(cands['koi_ror'])) & (~np.isnan(cands['koi_period'])) & (cands['koi_period'] < 365.25*5)

    return cands[sel]

def filter_quarter(qtr):
    """Takes a quarter array (for example, read by :func:`load_quarter`)
    and removes any objects not in the "EX" program or with bad log(g)
    or radius measurements.

    """
    sel = (qtr['program'] == 'EX') & (~np.isnan(qtr['logg'])) & (~np.isnan(qtr['rad']) & (qtr['logg'] > 0) & (qtr['rad'] > 0) & (qtr['cdpp12'] > 0))

    return qtr[sel]

def accumulate_cdpps(dir):
    """Reads all ``cdpp_quarterN.txt.bz2`` files in the given directory,
    accumulating the RMS CDPP values for all targets that have been
    observed in any quarter.  Returns a dictionary from kepler ID's to
    arrays with elements

      * ``ID``
      * ``cdpp12``
      * ``program``
      * ``logg``
      * ``rad``
      * ``nquarter``

    """
    cdpp_map = {}

    citem_template = np.zeros(1, dtype=[('ID', np.int),
                                        ('cdpp12', np.float),
                                        ('program', np.str, 20),
                                        ('logg', np.float),
                                        ('rad', np.float),
                                        ('nquarter', np.int)])

    for cfile in glob.glob(op.join(dir, 'cdpp_quarter*.txt.bz2')):
        with bz2.BZ2File(cfile, 'r') as inp:
            qdata = filter_quarter(load_quarter(inp))

        for qitem in qdata:
            id = qitem['ID']

            if cdpp_map.has_key(id):
                old_item = cdpp_map[id]
                new_cdpp = np.sqrt((np.square(old_item['cdpp12'])*old_item['nquarter'] + np.square(qitem['cdpp12']))/(old_item['nquarter'] + 1))
                new_quarter = old_item['nquarter'] + 1

                new_item = old_item.copy()
                new_item['cdpp12'] = new_cdpp
                new_item['nquarter'] = new_quarter

                cdpp_map[id] = new_item
            else:
                item = citem_template.copy()
                for n in qitem.dtype.names:
                    item[n] = qitem[n]
                item['nquarter'] = 1
                cdpp_map[id] = item

    return cdpp_map

def logg_to_mass(logg, radius):
    """Returns the mass in solar masses from the given log10(g) in cm/s/s
    and radius in solar radii.

    """
    return np.square(radius)*0.00003644310382*np.exp(2.302585093*logg)

def cdpps_to_snr0s(cdpps):
    """Takes a map from ID's to CDPP and returns a map from ID to SNR for
    a 1 REarth planet in a 1 Year orbit (this is the so-called SNR0
    quantity used in the posterior).

    """

    snr0s = {}

    for id, row in cdpps.items():
        cdpp = row['cdpp12']
        qs = row['nquarter']
        r = row['rad']
        m = logg_to_mass(row['logg'], row['rad'])

        depth = 83.790/(r*r) # ppm of 1 REarth
        tdur = 0.5858212976*r/m**(1.0/3.0) # duration in 12-hours
        ntr = 0.25*qs # Number of transits at 1 Yr orbit.

        snr0s[id] = (depth/cdpp*np.sqrt(tdur*ntr))[0]

    return snr0s

def save_stellar_properties(cdpp_map, snr0_map, file):
    """Writes the stellar properties to the given file object in
    tab-separated columns.  The columns are

     * Kepler ID
     * Radius (RSun)
     * Mass (MSun)
     * SNR0 (SNR of 1 REarth in 1 Year orbit)

    """

    file.write('Kepler ID\tRadius\tMass\tSNR0\n')

    for id, qitem in cdpp_map.items():
        file.write('{0:d}\t{1:g}\t{2:g}\t{3:g}\n'.format(id, qitem['rad'][0], logg_to_mass(qitem['logg'][0], qitem['rad'][0]), snr0_map[id]))

def save_candidate_properties(cands, snr0_map, cdpp_map, file):
    """Writes the properties for each candidate to the given file in
    tab-separated columns.  The columns are

     * Kepler ID
     * Period (Years)
     * Radius (REarth)
     * Stellar Radius (RSun)
     * Stellar Mass (MSun)
     * SNR0 (SNR of 1 REarth in 1 Year orbit)

    """

    file.write('Kepler ID\tPeriod\tRadius\tStellar Radius\tStellar Mass\tSNR0\n')

    for cand in cands:
        if cdpp_map.has_key(cand['kepid']) and snr0_map.has_key(cand['kepid']):
            id = cand['kepid']
            p = cand['koi_period']/365.25
            rs = cdpp_map[id]['rad'][0]
            r = 109.1665359*cand['koi_ror']*rs
            m = logg_to_mass(cdpp_map[id]['logg'][0], cdpp_map[id]['rad'][0])
            snr0 = snr0_map[id]

            file.write('{0:d}\t{1:g}\t{2:g}\t{3:g}\t{4:g}\t{5:g}\n'.format(id, p, r, rs, m, snr0))
