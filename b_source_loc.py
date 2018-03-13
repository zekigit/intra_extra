from pyface.qt import QtGui, QtCore
#%matplotlib qt
import mne
import glob
import sys
import numpy as np
import os.path as op
import pandas as pd
# import matplotlib
# matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
from mayavi import mlab
from scipy.spatial.distance import euclidean
from intra_extra_fx import find_stim_coords, get_stim_params, plot_source_space, save_results
from intra_extra_info import study_path, subjects_dir
import re
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import nibabel as nib
pd.set_option('display.expand_frame_repr', False)


def make_vol_fwd(epochs, fs_subj, study_path, subjects_dir):
    cond = epochs.info['description']

    bem_fname = op.join(subjects_dir, fs_subj, 'bem', '%s-bem-sol.fif' % fs_subj)
    trans_fname = op.join(study_path, 'source_stim', subj, 'source_files', img_type, '%s_fid-trans.fif' % fs_subj)

    mri_file = op.join(subjects_dir, subj, 'mri', 'T1.mgz')
    v_source = mne.setup_volume_source_space(subject=fs_subj, mri=mri_file, bem=bem_fname, subjects_dir=subjects_dir,
                                             pos=5.)

    mne.viz.plot_alignment(epochs.info, trans=trans_fname, subject=fs_subj, subjects_dir=subjects_dir,
                           surfaces=['seghead', 'brain'], bem=bem_fname, coord_frame='mri') #, src=v_source



    fwd = mne.make_forward_solution(epochs.info, trans_fname, v_source, bem_fname,
                                    mindist=5.0,  # ignore sources<=5mm from innerskull
                                    meg=False, eeg=True,
                                    n_jobs=3)

    mne.write_forward_solution(op.join(study_path, 'source_stim', subj, 'source_files', img_type,
                               '%s_vol_source_space_5mm-fwd.fif' % subj), fwd)
    return fwd


def vol_source_loc(evoked, fwd, cov, fs_subj, method='sLORETA'):
    cond = evoked.info['description']
    stim_coords = find_stim_coords(cond, subj, study_path)
    stim_params = get_stim_params(cond)

    evo_crop = evoked.crop(-0.005, 0.005)

    inv = mne.minimum_norm.make_inverse_operator(evoked.info, fwd, cov, loose=1, depth=None)
    snr = 30.
    lambda2 = 1. / snr ** 2
    stc = mne.minimum_norm.apply_inverse(evo_crop, inv, lambda2, method=method, pick_ori=None)

    from nilearn.plotting import plot_stat_map
    from nilearn.image import index_img
    import nilearn as ni

    nii_fname = op.join(study_path, 'source_stim', subj, 'source_files', img_type, 'stc', '%s_%s_stc.nii' % (subj, cond))
    img = mne.save_stc_as_volume(nii_fname, stc.copy().crop(-0.003, 0.003), fwd['src'], dest='mri', mri_resolution=True)
    mri_file = op.join(subjects_dir, fs_subj, 'mri', 'T1.mgz')

    from nibabel.affines import apply_affine
    mri = nib.load(mri_file)

    img_dat = img.get_data()
    stim_loc_vox = np.unravel_index(img_dat.argmax(), img_dat.shape)
    t_max = stim_loc_vox[-1]

    img_max = index_img(img, t_max)

    loc_coords = apply_affine(mri.affine, stim_loc_vox[:3])
    dist = euclidean(stim_coords['scanner_RAS'], loc_coords)
    print('Distance: %0.1f mm' % dist)
    thr = np.percentile(img.get_data(), 99.9)
    st_map = plot_stat_map(img_max, mri_file, threshold=thr, display_mode='ortho', cmap=plt.cm.plasma,
                           cut_coords=loc_coords)

    # img_smo = ni.image.smooth_img(index_img(img, t_max), fwhm=50)
    # smo_dat = img_smo.get_data()
    # stim_loc_vox_smo = np.unravel_index(smo_dat.argmax(), smo_dat.shape)
    # loc_coords = apply_affine(mri.affine, stim_loc_vox_smo[:3])
    # dist = euclidean(stim_coords['scanner_RAS'], loc_coords)
    # print('Distance: %0.1f mm' % dist)
    #
    # thr = np.percentile(img_smo.get_data(), 99.99)
    # st_map = plot_stat_map(img_smo, mri_file, threshold=thr, display_mode='ortho', cmap=plt.cm.plasma,
    #                        cut_coords=loc_coords)
    # fname_fig = op.join(study_path, 'source_stim', subj, 'figures', 'distributed', '%s_%s_vol.pdf' % (subj, cond))
    # st_map.savefig(fname_fig)
    # plt.close()
    return dist


def dip_source_loc(evoked, cov,  fs_subj, study_path, plot=False):
    evo_crop = evoked.copy().crop(-0.005, 0.005)
    subj = fs_subj.strip('_an') if fs_subj.find('_an') > 0 else fs_subj
    img_type = 'anony' if fs_subj.find('_an') > 0 else 'orig'

    trans_fname = op.join(study_path, 'source_stim', subj, 'source_files', img_type, '%s_fid-trans.fif' % fs_subj)
    bem_fname = op.join(subjects_dir, fs_subj, 'bem', '%s-bem-sol.fif' % fs_subj)

    cond = eeg_epo.info['description']
    stim_coords = find_stim_coords(cond, subj, study_path)

    dip = mne.fit_dipole(evo_crop, cov, bem_fname, trans_fname, min_dist=5, n_jobs=3)[0]

    import mne.transforms as tra
    from scipy import linalg
    trans = mne.read_trans(trans_fname)

    stim_point = stim_coords['surf_ori']  # get point for plot in mm
    #dip.pos[np.argmax(dip.gof)] = tra.apply_trans(surf_to_head, stim_coords['surf_ori']/1e3)  # check stim loc (apply affine in m)

    dip_surf = tra.apply_trans(trans['trans'], dip.pos[np.argmax(dip.gof)]) * 1e3  # transform from head to surface
    dist_surf = euclidean(dip_surf, stim_point)  # compute distance in surface space
    print(dist_surf)

    if plot:
        # Plot the result in 3D brain with the MRI image.
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        dip.plot_locations(trans_fname, fs_subj, subjects_dir, mode='orthoview', coord_frame='mri', ax=ax, show_all=True,
                           idx='gof')
        ax.scatter(stim_point[0], stim_point[1], stim_point[2])
        ax.plot([stim_point[0], -128], [stim_point[1], stim_point[1]], [stim_point[2], stim_point[2]], color='g')
        ax.plot([stim_point[0], stim_point[0]], [stim_point[1], -128], [stim_point[2], stim_point[2]], color='g')
        ax.plot([stim_point[0], stim_point[0]], [stim_point[1], stim_point[1]], [stim_point[2], -128], color='g')
        ax.text2D(0.05, 0.90, 'distance: %i mm \nstim coords = %0.1f %0.1f %0.1f' % (dist_surf, stim_point[0], stim_point[1], stim_point[2]),
                  transform=ax.transAxes)
        red_patch = mpatches.Patch(color='red')
        green_patch = mpatches.Patch(color='green')
        fig.legend(handles=[red_patch, green_patch], labels=['dipole', 'electrode'])

        fig.savefig(op.join(study_path, 'source_stim', subj, 'figures', 'dipole', '%s_dip_15mm.png' % cond))
        plt.close()

        from mne.viz import plot_dipole_locations, plot_dipole_amplitudes
        plot_dipole_amplitudes(dip)
        plot_dipole_locations(dip, trans, subj, subjects_dir=subjects_dir)
    return dist_surf


if __name__ == '__main__':
    subj = 'S3'
    img_type = 'orig'
    fs_subj = subj + '_an' if img_type == 'anony' else subj

    epo_path = op.join(study_path, 'source_stim', subj, 'epochs', 'fif')
    conds = glob.glob(epo_path + '/*-epo.fif')

    fwd_fname = op.join(study_path, 'source_stim', subj, 'source_files', img_type,
                        '%s_vol_source_space_5mm-fwd.fif' % fs_subj)

    for cond_fname in conds:
        eeg_epo = mne.read_epochs(cond_fname)
        eeg_epo.interpolate_bads(reset_bads=True)

        if not op.isfile(fwd_fname):
            fwd = make_vol_fwd(eeg_epo, fs_subj, study_path, subjects_dir)
        else:
            fwd = mne.read_forward_solution(fwd_fname)

        # filter??
        # epo_base = eeg_epo.copy().crop(-0.5, -0.05)
        # epo_base.filter(None, 40)
        # cov = mne.compute_covariance(epo_base, method='shrunk', tmin=-0.45, tmax=-0.15)  # use method='auto' for final computation

        cov = mne.compute_covariance(eeg_epo, method='shrunk', tmin=-0.5, tmax=-0.3)  # use method='auto' for final computation
        evoked = eeg_epo.average()

        # evoked.plot(spatial_colors=True)
        # evoked.plot_topomap(np.linspace(-0.005, 0.005, 11))

        dist_dis = vol_source_loc(evoked, fwd, cov, fs_subj)
        dist_dip = dip_source_loc(evoked, cov,  fs_subj, study_path, plot=False)

        cond = eeg_epo.info['description']
        stim_params = get_stim_params(cond)

        save_results(subj, study_path, stim_params, evoked, img_type, dist_dis, dist_dip)
