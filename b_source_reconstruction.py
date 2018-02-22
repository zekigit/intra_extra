import mne
import glob
import sys
import numpy as np
import os.path as op
import pandas as pd
import matplotlib.pyplot as plt
from mayavi import mlab
from scipy.spatial.distance import euclidean
from intra_extra_info import study_path, subjects, subjects_dir, bad_chans
from intra_extra_fx import find_stim_coords, get_stim_params
import re
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
pd.set_option('display.expand_frame_repr', False)


def make_fwd_solution(cond_fname, subj, study_path):
    eeg_epo = mne.read_epochs(cond_fname)

    trans_fname = op.join(study_path, 'source_stim', subj, 'source_files', '%s-trans.fif' % subj)
    src_fname = op.join(study_path, 'source_stim', subj, 'source_files', '%s-oct5-src.fif' % subj)
    bem_fname = op.join(study_path, 'freesurfer_subjects', subj, 'bem', '%s-bem-sol.fif' % subj)

    fwd = mne.make_forward_solution(eeg_epo.info, trans_fname, src_fname, bem_fname)

    # trans = mne.read_trans(trans_fname)
    # mne.viz.plot_alignment(eeg_epo.info, trans, subject='%s' % subj, subjects_dir=op.join(study_path, 'freesurfer_subjects'))
    # mlab.show()

    mne.write_forward_solution(op.join(study_path, 'source_stim', subj, 'source_files', '%s-fwd.fif' % subj), fwd, overwrite=True)
    return fwd


def source_loc(cond_fname, subj, fwd):
    eeg_epo = mne.read_epochs(cond_fname)
    cov = mne.compute_covariance(eeg_epo, method='shrunk', tmin=-0.5, tmax=-0.3)
    inv = mne.minimum_norm.make_inverse_operator(eeg_epo.info, fwd, cov, loose=0.2, depth=0.8)

    evoked = eeg_epo.average()

    method = "sLORETA"
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
    stim_coords = stim_coords['surf']

    stim_params = get_stim_params(cond)

    hemi = 'lh' if stim_params['is_left'] else 'rh'
    contact_nr = int(re.findall('\d+', stim_params['ch'])[0])
    view = 'medial' if contact_nr < 5 else 'lateral'


    vertno_max, time_max = stc.get_peak(hemi=hemi)
    stc_max = np.max(stc.data)

    brain_inf = stc.plot(surface='inflated', hemi=hemi, subjects_dir=subjects_dir,
                         clim=dict(kind='value', lims=[0, stc_max*0.75, stc_max]),
                         initial_time=time_max, time_unit='s', alpha=1, background='w', foreground='k')

    brain_inf.add_foci(vertno_max, coords_as_verts=True, hemi=hemi, color='blue',
                       scale_factor=0.8)

    brain_inf.add_foci(stim_coords, map_surface='pial', hemi=hemi, color='green', scale_factor=0.8)

    brain_inf.show_view(view)  # mlab.view(130, 90)
    fig_fname = op.join(study_path, 'source_stim', subj, 'figures', '%s_inf_foci.eps' % cond)
    brain_inf.save_image(fig_fname)
    mlab.clf()
    #mlab.show()


    # brain_pial = stc.plot(surface='pial', hemi='lh', subjects_dir=subjects_dir,
    #                      clim=dict(kind='value', lims=[0, stc_max*0.75, stc_max]),
    #                      initial_time=time_max, time_unit='s', alpha=1, background='w', foreground='k')
    #
    # brain_pial.add_foci(vertno_max, coords_as_verts=True, hemi='lh', color='blue',
    #                    scale_factor=1.5)
    #
    # brain_pial.add_foci(stim_coords, hemi='lh', color='green', scale_factor=1.5)
    #
    # brain_pial.show_view('medial')  # mlab.view(130, 90)
    # fig_fname = op.join(study_path, 'source_stim', subj, 'figures', '%s_pial_foci.eps' % cond)
    # brain_pial.save_image(fig_fname)
    # mlab.show()


def dipole_source_loc(cond_fname, subj, study_path):
    eeg_epo = mne.read_epochs(cond_fname)
    cov = mne.compute_covariance(eeg_epo, method='shrunk', tmin=-0.5, tmax=-0.3)

    evoked = eeg_epo.average()

    evo_crop = evoked.crop(-0.001, 0.001)
    trans_fname = op.join(study_path, 'source_stim', subj, 'source_files', '%s-trans.fif' % subj)
    bem_fname = op.join(study_path, 'freesurfer_subjects', subj, 'bem', '%s-bem-sol.fif' % subj)

    cond = eeg_epo.info['description']
    stim_coords = find_stim_coords(cond, subj, study_path)
    stim_coords_sf = stim_coords['surf']
    seeg_ch_info = pd.read_csv(op.join(study_path, 'physio_data', subj, 'chan_info', '%s_seeg_ch_info.csv' % cond))

    # Fit a dipole
    dip = mne.fit_dipole(evoked, cov, bem_fname, trans_fname)[0]
    dist = euclidean(stim_coords['head'], dip.pos[0]*1e3)

    trans = mne.read_trans(trans_fname)

    # Plot the result in 3D brain with the MRI image.

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    dip.plot_locations(trans_fname, 'S5', subjects_dir, mode='orthoview', coord_frame='mri', ax=ax, show_all=False)
    ax.scatter(stim_coords_sf[0], stim_coords_sf[1], stim_coords_sf[2])
    ax.plot([stim_coords_sf[0], -128], [stim_coords_sf[1], stim_coords_sf[1]], [stim_coords_sf[2], stim_coords_sf[2]], color='g')
    ax.plot([stim_coords_sf[0], stim_coords_sf[0]], [stim_coords_sf[1], -128], [stim_coords_sf[2], stim_coords_sf[2]], color='g')
    ax.plot([stim_coords_sf[0], stim_coords_sf[0]], [stim_coords_sf[1], stim_coords_sf[1]], [stim_coords_sf[2], -128], color='g')
    red_patch = mpatches.Patch(color='red')
    green_patch = mpatches.Patch(color='green')
    fig.legend(handles=[red_patch, green_patch], labels=['dipole', 'electrode'])
    ax.text2D(0.05, 0.90, 'distance: %i mm \n stim coords = %i %i %i' % (dist, stim_coords_sf[0], stim_coords_sf[1], stim_coords_sf[2]),
              transform=ax.transAxes)

    fig.savefig(op.join(study_path, 'source_stim', subj, 'figures', '%s_dip.jpg' % cond))


subj = 'S5'
study_path = '/Volumes/MAXTOR'

if __name__ == '__main__':
    subj = sys.argv[1]
    study_path = sys.argv[2]

    epo_path = op.join(study_path, 'source_stim', subj, 'epochs')
    conds = glob.glob(epo_path + '/*-epo.fif')

    fwd_fname = op.join(study_path, 'source_stim', subj, 'source_files', '%s-fwd.fif' % subj)
    if not op.isfile(fwd_fname):
        fwd = make_fwd_solution(conds[0], subj, study_path)
    else:
        fwd = mne.read_forward_solution(fwd_fname)

    for cond_fname in conds:
        #source_loc(cond_fname, subj, fwd)
        dipole_source_loc(cond_fname, subj, study_path)
