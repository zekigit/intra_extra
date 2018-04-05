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


def make_fwd_solution(cond_fname, fs_subj, study_path):
    eeg_epo = mne.read_epochs(cond_fname)
    img_type = 'anony' if fs_subj.find('_an') > 0 else 'orig'
    subj = fs_subj.strip('_an')

    trans_fname = op.join(study_path, 'source_stim', subj, 'source_files', img_type,'%s_fid-trans.fif' % fs_subj)
    src_fname = op.join(study_path, 'source_stim', subj, 'source_files', img_type, '%s-oct5-src.fif' % fs_subj)
    bem_fname = op.join(subjects_dir, fs_subj, 'bem', '%s-bem-sol.fif' % fs_subj)

    fwd = mne.make_forward_solution(eeg_epo.info, trans_fname, src_fname, bem_fname)

    trans = mne.read_trans(trans_fname)
    mne.viz.plot_alignment(eeg_epo.info, trans, subject=subj, surfaces=['head', 'brain'], subjects_dir=subjects_dir)
    # mlab.show()

    mne.write_forward_solution(op.join(study_path, 'source_stim', subj, 'source_files', img_type,
                                       '%s-fwd.fif' % fs_subj), fwd, overwrite=True)
    return fwd


def source_loc(evoked, cov, fwd, subj, img_type, study_path, plot=False):
    cond = evoked.info['description']
    stim_coords = find_stim_coords(cond, subj, study_path)
    stim_params = get_stim_params(cond)

    inv = mne.minimum_norm.make_inverse_operator(evoked.info, fwd, cov, loose=0.2, depth=0.8)
    method = "dSPM"
    snr = 300.
    lambda2 = 1. / snr ** 2
    stc = mne.minimum_norm.apply_inverse(evoked, inv, lambda2, method=method, pick_ori=None)
    stc.save(op.join(study_path, 'source_stim', subj, 'source_files', img_type,'stc', '%s_%s' %(subj, cond)))

    # plot_source_space(subj, study_path, subjects_dir)

    hemi = 'lh' if stim_params['is_left'] else 'rh'
    contact_nr = int(re.findall('\d+', stim_params['ch'])[0])
    view = 'medial' if contact_nr < 5 else 'lateral'  # improve view selection

    vertno_max, time_max = stc.get_peak(hemi=hemi)
    stc_max = np.max(stc.data)

    if plot:
        brain_inf = stc.plot(surface='inflated', hemi=hemi, subjects_dir=subjects_dir, subject=fs_subj,
                             clim=dict(kind='value', lims=[0, stc_max*0.75, stc_max]),
                             initial_time=time_max, time_unit='s', alpha=1, background='w', foreground='k')

        brain_inf.add_foci(vertno_max, coords_as_verts=True, hemi=hemi, color='blue',
                           scale_factor=0.8)

        brain_inf.add_foci(stim_coords['surf_ori'], map_surface='pial', hemi=hemi, color='green', scale_factor=0.8)

        brain_inf.show_view(view)  # mlab.view(130, 90)
        fig_fname = op.join(study_path, 'source_stim', subj, 'figures', 'distributed', '%s_inf_foci.eps' % cond)
        #brain_inf.save_image(fig_fname)
        mlab.show()
        #mlab.clf()

    max_mni = mne.vertex_to_mni(vertno_max, hemis=0, subject=subj, subjects_dir=subjects_dir)  # mni coords of vertex (ojo vertex resolution ico-5)
    stim_mni = stim_coords['mni']  # mni coords of stimulation
    dist_mni = euclidean(max_mni, stim_mni)

    print(dist_mni)
    return dist_mni

    # brain_pial = stc.plot(surface='pial', hemi=hemi, subjects_dir=subjects_dir,
    #                      clim=dict(kind='value', lims=[0, stc_max*0.75, stc_max]),
    #                      initial_time=time_max, time_unit='s', alpha=1, background='w', foreground='k')
    #
    # brain_pial.add_foci(vertno_max, coords_as_verts=True, hemi=hemi, color='blue',
    #                    scale_factor=1)
    #
    # brain_pial.add_foci(stim_coords['surf'], hemi='lh', color='green', scale_factor=1)
    #
    # brain_pial.show_view('medial')  # mlab.view(130, 90)
    # fig_fname = op.join(study_path, 'source_stim', subj, 'figures', 'distributed', '%s_pial_foci.eps' % cond)
    # brain_pial.save_image(fig_fname)
    # mlab.show()



def vol_source_loc(evoked, cov, fs_subj, study_path, plot=True):
    cond = evoked.info['description']

    bem_fname = op.join(subjects_dir, fs_subj, 'bem', '%s-bem-sol.fif' % fs_subj)
    trans_fname = op.join(study_path, 'source_stim', subj, 'source_files', img_type, '%s_static-trans.fif' % fs_subj)


    mri_file = op.join(subjects_dir, subj, 'mri', 'T1.mgz')
    v_source = mne.setup_volume_source_space(subject=fs_subj, mri=mri_file, bem=bem_fname, subjects_dir=subjects_dir,
                                             pos=2.)

    fwd = mne.make_forward_solution(evoked.info, trans_fname, v_source, bem_fname,
                                mindist=5.0,  # ignore sources<=5mm from innerskull
                                meg=False, eeg=True,
                                n_jobs=2)

    mne.write_forward_solution(op.join(study_path, 'source_stim', subj, 'source_files', '%s_vol_source_space_2mm'),
                               fwd)

    inv = mne.minimum_norm.make_inverse_operator(evoked.info, fwd, cov, loose=1, depth=1)
    method = "sLORETA"
    snr = 30.
    lambda2 = 1. / snr ** 2
    stc = mne.minimum_norm.apply_inverse(evoked, inv, lambda2, method=method, pick_ori=None)

    stc.crop(-0.005, 0.005)

    from nilearn.plotting import plot_stat_map
    from nilearn.image import index_img
    import nilearn as ni

    nii_fname = op.join(study_path, 'source_stim', subj, 'source_files', img_type, 'stc', '%s_%s_stc.nii' % (subj, cond))
    img = mne.save_stc_as_volume(nii_fname, stc.copy().crop(-0.003, 0.003), fwd['src'], dest='mri')

    t0_img = index_img(img, 3)
    act_coord = ni.plotting.find_xyz_cut_coords(t0_img)
    #act_coord = [57, 122, 87]
    thr = np.percentile(t0_img.get_data(), 99.9)
    st_map = plot_stat_map(t0_img, mri_file, threshold=thr, display_mode='ortho', cmap=plt.cm.plasma)
    st_map.add_markers(np.reshape(act_coord, (-1, 3)), marker_color='g', marker_size=50)
    fname_fig = op.join(study_path, 'source_stim', subj, 'figures', 'volume', '%s_%s_st_map.pdf' % (subj, cond))
    st_map.savefig(fname_fig)

    dist_vol = euclidean()


def dipole_source_loc(evoked, cov,  fs_subj, study_path, plot=False):
    evo_crop = evoked.copy().crop(-0.005, 0.005)
    subj = fs_subj.strip('_an') if fs_subj.find('_an') > 0 else fs_subj
    img_type = 'anony' if fs_subj.find('_an') > 0 else 'orig'

    seeg_ch_info = pd.read_csv(op.join(study_path, 'physio_data', subj, 'chan_info', '%s_seeg_ch_info.csv' % subj))

    # ori_to_an_fname = op.join(study_path, 'freesurfer_subjects', subj, 'mri', 'ori_to_an_trans.lta')  # mri to mri
    # ori_to_an_trans = np.genfromtxt(ori_to_an_fname, skip_header=8, skip_footer=18)

    trans_fname = op.join(study_path, 'source_stim', subj, 'source_files', img_type, '%s_static-trans.fif' % fs_subj)
    bem_fname = op.join(subjects_dir, fs_subj, 'bem', '%s-bem-sol.fif' % fs_subj)

    cond = eeg_epo.info['description']
    stim_coords = find_stim_coords(cond, subj, study_path)

    # Fit a dipole
    dip = mne.fit_dipole(evo_crop, cov, bem_fname, trans_fname, min_dist=5, n_jobs=2)[0]

    from mne.beamformer import rap_music
    dip = rap_music(evo_crop, fwd, cov, n_dipoles=1, return_residual=True, verbose=True)[0][0]


    import mne.transforms as tra
    from scipy import linalg
    trans = mne.read_trans(trans_fname)

    surf_to_head = linalg.inv(trans['trans'])

    # stim_point = tra.apply_trans(surf_to_head, stim_coords['surf']/1e3)
    stim_point = stim_coords['surf_ori']  # get point for plot in mm
    #dip.pos[np.argmax(dip.gof)] = tra.apply_trans(surf_to_head, stim_coords['surf_ori']/1e3)  # check stim loc (apply affine in m)

    #stim_point = tra.apply_trans(ori_to_an_trans, stim_point)

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
    subj = 'S5'
    img_type = 'orig'
    fs_subj = subj + '_an' if img_type == 'anony' else subj

    epo_path = op.join(study_path, 'source_stim', subj, 'epochs', 'fif')
    conds = glob.glob(epo_path + '/*-epo.fif')

    fwd_fname = op.join(study_path, 'source_stim', subj, 'source_files', img_type, '%s-fwd.fif' % fs_subj)

    if not op.isfile(fwd_fname):
        fwd = make_fwd_solution(conds[0], fs_subj, study_path)
    else:
        fwd = mne.read_forward_solution(fwd_fname)

    for cond_fname in conds:
        eeg_epo = mne.read_epochs(cond_fname)
        eeg_epo.interpolate_bads(reset_bads=True)
        # eeg_epo.filter(None, 40)
        cov = mne.compute_covariance(eeg_epo, method='shrunk', tmin=-0.5, tmax=-0.3)  # use method='auto' for final computation
        evoked = eeg_epo.average()

        evoked.plot(spatial_colors=True)
        mne.viz.plot_evoked_topo(evoked)
        evoked.plot_topomap(np.linspace(-0.005, 0.005, 11))

        dist_dis = source_loc(evoked, cov, fwd, subj, img_type, study_path, plot=False)
        dist_dip = dipole_source_loc(evoked, cov,  fs_subj, study_path, plot=True)

        cond = eeg_epo.info['description']
        stim_params = get_stim_params(cond)

        save_results(subj, study_path, stim_params, evoked, dist_dis, dist_dip, img_type)




