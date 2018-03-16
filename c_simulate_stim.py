import mne
import glob
import numpy as np
import os.path as op
import matplotlib.pyplot as plt
from mne.time_frequency import fit_iir_model_raw
from mne.simulation import simulate_sparse_stc, simulate_evoked
from intra_extra_info import study_path

subj = 'S5'

fwd_fname = op.join(study_path, 'source_stim', subj, 'source_files', 'orig',
                    '%s-fwd.fif' % subj)

epo_path = op.join(study_path, 'source_stim', subj, 'epochs', 'fif')
conds = glob.glob(epo_path + '/*-epo.fif')

cond_fname = conds[0]
epochs = mne.read_epochs(cond_fname)
epochs.interpolate_bads(reset_bads=True)

fwd = mne.read_forward_solution(fwd_fname)
fwd1 = mne.convert_forward_solution(fwd, force_fixed=True)
cov = mne.compute_covariance(epochs, method='shrunk', tmin=-0.5, tmax=-0.3)  # use method='auto' for final computation

evoked = epochs.average()
info = evoked.info


def data_fun(times):
    data = np.hstack((np.repeat(0, 1000), 5e-6, 5e-6, -5e-6, -5e-6, np.repeat(0, 996)))
    return data

times = np.linspace(-0.5, 0.5, 2000)
stc = simulate_sparse_stc(fwd1['src'], n_dipoles=1, times=times,
                          random_state=42, labels=None, data_fun=data_fun)

info['sfreq'] = 2000
#iir_filter = fit_iir_model_raw(raw, order=5, picks=picks, tmin=60, tmax=180)[1]
nave = 30  # simulate average of 100 epochs
evoked_sim = simulate_evoked(fwd, stc, info, cov, nave=nave, use_cps=True,
                             iir_filter=None)
evoked_sim.resample(1000)
# evoked_sim.filter(None, 250)

fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
evoked_sim.plot(spatial_colors=True, axes=axes[0])
evoked.plot(spatial_colors=True, axes=axes[1])



