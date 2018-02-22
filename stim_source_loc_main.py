import mne
import glob
import numpy as np
import os.path as op
import pandas as pd
import matplotlib.pyplot as plt
from mayavi import mlab
from scipy.spatial.distance import euclidean
from intra_extra_info import study_path, subjects, subjects_dir, bad_chans
from intra_extra_fx import find_stim_coords

pd.set_option('display.expand_frame_repr', False)


def load_data(subj, cond_fname):
    eeg_raw = mne.io.read_raw_edf(cond_fname + '_epochs.edf', preload=True, stim_channel=-1)
    eeg_raw._data[:-1, :] /= 1e6  # scale to volts
    events = mne.find_events(eeg_raw, uint_cast=True, consecutive=False)
    events[:, 2] = 1
    eeg_epo = mne.Epochs(eeg_raw, events, event_id={'stim': 1}, tmin=-0.5, tmax=0.5, baseline=(-0.5, -0.3))
    dig_fname = op.join(study_path, 'physio_data', subj, 'chan_info', '%s_egi_digitalization.hpts' % subj)
    montage = mne.channels.read_montage(dig_fname, unit='mm')
    # montage.plot(kind='3d')

    eeg_epo.set_montage(montage, set_dig=True)
    eeg_epo.info['description'] = op.split(cond_fname)[-1]
    # eeg_epo.plot_sensors(kind='3d')
    return eeg_epo


def preprocess(eeg_epo):
    eeg_epo.plot(n_channels=32, scalings={'eeg': 6e-4})
    bads = bad_chans[subj]
    eeg_epo.info['bads'] = bads
    eeg_epo.set_eeg_reference('average', projection=True)
    eeg_epo.apply_proj()
    eeg_epo.filter(0.1, None, method='iir', iir_params=None)

    epo_fname = op.join(study_path, 'source_stim', 'epochs', '%s-epo.fif' % eeg_epo.info['description'])
    eeg_epo.save(epo_fname)
    return eeg_epo


def make_fwd_solution(cond_fname, subj, study_path):
    eeg_epo = mne.read_epochs(cond_fname + '-epo.fif')

    trans_fname = op.join(study_path, 'source_stim', 'images', subj, '%s_coreg-trans.fif' % subj)
    src_fname = op.join(study_path, 'source_stim', 'images', subj, '%s-oct5-src.fif' % subj)
    bem_fname = op.join(study_path, 'freesurfer_subjects', '%s' % subj, 'bem', '%s-bem-sol.fif' % subj)

    fwd = mne.make_forward_solution(eeg_epo.info, trans_fname, src_fname, bem_fname)

    mne.viz.plot_bem(subject='S1', subjects_dir=subjects_dir, src=src_fname)
    mne.viz.plot_bem(subject='S3', subjects_dir=subjects_dir)

    # trans = mne.read_trans(trans_fname)
    # mne.viz.plot_alignment(eeg_epo.info, trans, subject='%s' % subj, subjects_dir=op.join(study_path, 'freesurfer_subjects'))
    # mlab.show()

    # mne.scale_bem('%s_coreg' % subj, 'bem-sol', subjects_dir=subjects_dir)

    mne.write_forward_solution(op.join(study_path, 'source_stim', 'images', subj, '%s-fwd.fif' % subj), fwd, overwrite=True)
    return fwd


def source_loc(cond_fname, subj, fwd):
    eeg_epo = mne.read_epochs(cond_fname + '-epo.fif')
    cov = mne.compute_covariance(eeg_epo, method='shrunk', tmin=-0.5, tmax=-0.3)
    inv = mne.minimum_norm.make_inverse_operator(eeg_epo.info, fwd, cov, loose=0.2, depth=0.8)

    evoked = eeg_epo.average()

    method = "dSPM"
    snr = 3.
    lambda2 = 1. / snr ** 2
    stc = mne.minimum_norm.apply_inverse(evoked, inv, lambda2, method=method, pick_ori=None)

    # from surfer import Brain  # noqa
    # src_fname = op.join(study_path, 'source_stim', 'images', subj, '%s-oct5-src.fif' % subj)
    # src = mne.read_source_spaces(src_fname, patch_stats=True)
    #
    # brain = Brain(subj, 'lh', 'inflated', subjects_dir=subjects_dir)
    # surf = brain.geo['lh']
    #
    # vertidx = np.where(src[0]['inuse'])[0]
    #
    # mlab.points3d(surf.x[vertidx[:-1]], surf.y[vertidx[:-1]],
    #               surf.z[vertidx[:-1]], color=(1, 1, 0), scale_factor=1.5)
    # mlab.show()

    cond = eeg_epo.info['description']
    stim_coords = find_stim_coords(cond, subj, study_path)

    vertno_max, time_max = stc.get_peak(hemi='lh')
    stc_max = np.max(stc.data)

    brain_inf = stc.plot(surface='inflated', hemi='lh', subjects_dir=subjects_dir,
                         clim=dict(kind='value', lims=[0, stc_max*0.75, stc_max]),
                         initial_time=time_max, time_unit='s', alpha=1, background='w', foreground='k')

    brain_inf.add_foci(vertno_max, coords_as_verts=True, hemi='lh', color='blue',
                       scale_factor=0.8)

    brain_inf.add_foci(stim_coords, map_surface='inflated', hemi='lh', color='green', scale_factor=0.8)

    brain_inf.show_view('lateral')  # mlab.view(130, 90)
    fig_fname = op.join(study_path, 'source_stim', 'figures', '%s_inf_foci.eps' % cond)
    brain_inf.save_image(fig_fname)
    mlab.show()








    # brain_pial = stc.plot(surface='pial', hemi='lh', subjects_dir=subjects_dir,
    #                       clim=dict(kind='value', lims=[0, stc_max*0.75, stc_max]),
    #                       initial_time=time_max, time_unit='s')
    #
    # brain_pial.add_foci(vertno_max, coords_as_verts=True, hemi='lh', color='blue',
    #                     scale_factor=0.8)
    #
    # brain_pial.add_foci(stim_coords, map_surface='pial', hemi='lh', color='green', scale_factor=0.8)
    #
    # brain_pial.show_view('lateral') # mlab.view(130, 90)
    # mlab.show()
    #
    # surf = brain_pial.geo['lh']
    # pk_coords = [surf.x[vertno_max], surf.y[vertno_max], surf.z[vertno_max]]
    #
    # pk_coors = mne.vertex_to_mni(vertno_max, 0, subject=subj, subjects_dir=subjects_dir)
    # loc_dist = euclidean(stim_coords, pk_coords)
    # print loc_dist

subj = subjects[3]

epo_path = op.join(study_path, 'source_stim', 'epochs')
conds = glob.glob(epo_path + '/*.edf')
cond_fname = conds[0].replace('_epochs.edf', '')


if '__name__' == '__main__':
    eeg_epo = load_data(subj, cond_fname)
    eeg_epo = preprocess(eeg_epo)

    fwd_fname = op.join(study_path, 'source_stim', 'images', subj, '%s-fwd.fif' % subj)

    if not op.isfile(fwd_fname):
        fwd = make_fwd_solution(cond_fname, subj, study_path)
    else:
        fwd = mne.read_forward_solution(fwd_fname)

    for c in conds:
        cond_fname = c.replace('_epochs.edf', '')
        source_loc(cond_fname, subj, fwd)



