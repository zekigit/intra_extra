from pyface.qt import QtGui, QtCore
import mne
import glob
import numpy as np
import os.path as op
import matplotlib.pyplot as plt
from mne.time_frequency import fit_iir_model_raw
from mne.simulation import simulate_sparse_stc, simulate_evoked
from mne.viz import plot_sparse_source_estimates
from intra_extra_info import study_path
from intra_extra_fx import epochs_fine_alignment, find_stim_coords
from egi_filters import btw_s1, btw_s2
from scipy import signal

subj = 'S5'
subjects_dir = op.join(study_path, 'freesurfer_subjects')

fwd_fname = op.join(study_path, 'source_stim', subj, 'source_files', 'orig',
                    '%s-fwd.fif' % subj)

trans = op.join(study_path, 'source_stim', subj, 'source_files', 'orig', '%s_fid-trans.fif' % subj)

epo_path = op.join(study_path, 'source_stim', subj, 'epochs', 'fif')
conds = glob.glob(epo_path + '/*-epo.fif')

cond_fname = conds[0]

epochs = mne.read_epochs(cond_fname)
epochs.interpolate_bads(reset_bads=True)
epochs = epochs_fine_alignment(epochs)

cond = epochs.info['description']
stim_coords = find_stim_coords(cond, subj, study_path)

fwd = mne.read_forward_solution(fwd_fname)
fwd1 = mne.convert_forward_solution(fwd, force_fixed=True)
cov = mne.compute_covariance(epochs, method='shrunk', tmin=-0.5, tmax=-0.3)  # use method='auto' for final computation

evoked = epochs.average()
info = evoked.info


from nibabel.freesurfer.io import read_geometry
fname_surf = op.join(subjects_dir, subj, 'surf', 'lh.pial')
surf = read_geometry(fname_surf)


def closest_nodes(node, all_nodes, used_nodes):
    nodes = np.asarray(all_nodes[used_nodes])
    dist_2 = np.sum((nodes - node)**2, axis=1)
    used_vertnos = dist_2.argsort()[:1]
    vertnos = used_nodes[used_vertnos]
    return vertnos


closes_vertnos = closest_nodes(stim_coords['surf'], surf[0], fwd['src'][0]['vertno'])
stim_vert = surf[0][closes_vertnos]

pulse = 5e-6
def data_fun(times):
    data = np.hstack((np.repeat(0, 1000), -pulse, -pulse, pulse, pulse, np.repeat(0, 997)))
    return data

times = np.linspace(-0.5, 0.5, 2001)


lab = mne.Label(closes_vertnos, subject=subj,
                pos=stim_vert, hemi='lh', name='stim')


stc = simulate_sparse_stc(fwd1['src'], n_dipoles=1, times=times, location='center', subject=subj,
                          subjects_dir=subjects_dir, random_state=42, labels=[lab], data_fun=data_fun)


plot_sparse_source_estimates(fwd1['src'], stc, high_resolution=True)

fwd['src'].plot(trans=trans, subjects_dir=subjects_dir)

info['sfreq'] = 2000
#iir_filter = fit_iir_model_raw(raw, order=5, picks=picks, tmin=60, tmax=180)[1]
nave = 30  # simulate average of 100 epochs
evoked_sim = simulate_evoked(fwd, stc, info, cov, nave=nave, use_cps=True,
                             iir_filter=None)

evoked_ds = evoked_sim.copy()
evoked_ds.filter(None, 400, method='iir', iir_params=None)
evoked_ds.resample(1000)
# evoked_sim.filter(None, 250)

fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
evoked_ds.plot(spatial_colors=True, axes=axes[0])
evoked.plot(spatial_colors=True, axes=axes[1])


for ep in epochs:
    plt.plot(epochs.times, ep[10, :])

from scipy.signal import resample, decimate

srate = 8000.
t = 0.08
pulse_l = 0.0005
pulse_i = 2.
x = np.linspace(1, srate * t, srate * t)
times = np.linspace(-t/2, t/2, srate * t)


def make_dat(x, srate, pulse_l, pulse_i):
    pulse_samps = srate * pulse_l
    noise = np.random.normal(-0.1, 0.01, len(x))
    y = np.hstack((np.repeat(0., (len(x)/2)), np.repeat(-pulse_i, pulse_samps), np.repeat(pulse_i, pulse_samps),
                   np.repeat(0., (len(x)/2)-pulse_samps*2)))
    y += noise
    y_samp = y[::8]
    x_samp = x[::8]
    offset = np.random.uniform(0, 0., 100)

    filt = mne.filter.create_filter(y, srate, None, 400)
    # mne.viz.plot_filter(filt, srate)
    #y_f = np.convolve(filt, y, mode='same')

    y_f = np.convolve(btw_s1, y, mode='same')
    y_f = np.convolve(btw_s2, y_f, mode='same')
    y_f = y_f[::8]
    times_y_f = decimate(times, 8)

    return y, y_f, times_y_f


y, y_f, times_y_f = make_dat(times, srate, pulse_l, pulse_i)

plt.plot(times, y, '-o', color='blue')
plt.plot(times_y_f, y_f, '-', color='green')

#plt.plot(x_samp, y_f_samp, '-o', color='red', linewidth=2)
plt.xlim([-0.01, 0.01])
#plt.plot(x_samp, y_samp, color='blue')

pulse_is = np.linspace(0.1, 1, 30)

ys = list()
ys_f = list()
reps = 100

for i in pulse_is:
    rep_y = list()
    rep_yf = list()
    for r in range(reps):
        y, y_f, times_y_f = make_dat(times, srate, pulse_l, i)
        rep_y.append(y.min())
        rep_yf.append(y_f.min())
    ys.append(rep_y)
    ys_f.append(rep_yf)


ys = np.array(ys)
ys_f = np.array(ys_f)


from scipy.stats.kde import gaussian_kde
from sklearn import linear_model
x = np.linspace(-0.4, -0.1, 301)
fig, axes = plt.subplots(2, 1, sharey=False, sharex=False)
[axes[ix].hist(vals) for ix, tipo in enumerate([ys, ys_f]) for vals in tipo]
for ix, f in enumerate(ys_f):
    dens = gaussian_kde(f)
    dens_f = dens(x)
    axes[1].plot(x, dens_f, 'k', alpha=0.5)
axes[0].set_xlabel('original amplitude')
axes[1].set_xlabel('filtered amplitude')


l_reg = linear_model.LinearRegression()
l_reg.fit(ys.reshape(-1, 1), ys_f.reshape(-1,1))
rang = np.linspace(ys.min(), ys.max(), 50)
preds = l_reg.predict(rang.reshape(-1, 1))

plt.plot(ys.T, ys_f.T, '.')
plt.plot(rang, preds, 'k', linewidth=4)
plt.ylabel('filtered amplitude')
plt.xlabel('original amplitude')

from sklearn.externals import joblib
fname_model = op.join(study_path, 'source_stim', subj, 'source_files', '%s_filt_regr.skmod' % subj)
joblib.dump(l_reg, fname_model)

