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
from intra_extra_fx import find_stim_coords, get_stim_params, plot_source_space
import re
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import nibabel as nib
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


def source_loc(evoked, cov, fwd, subj, study_path, plot=True):
    cond = evoked.info['description']
    stim_coords = find_stim_coords(cond, subj, study_path)
    stim_params = get_stim_params(cond)

    inv = mne.minimum_norm.make_inverse_operator(evoked.info, fwd, cov, loose=0.2, depth=0.8)
    method = "dSPM"
    snr = 3.
    lambda2 = 1. / snr ** 2
    stc = mne.minimum_norm.apply_inverse(evoked, inv, lambda2, method=method, pick_ori=None)
    stc.save(op.join(study_path, 'source_stim', subj, 'source_files', 'stc', '%s_%s' %(subj, cond)))

    # plot_source_space(subj, study_path, subjects_dir)

    hemi = 'lh' if stim_params['is_left'] else 'rh'
    contact_nr = int(re.findall('\d+', stim_params['ch'])[0])
    view = 'medial' if contact_nr < 5 else 'lateral'  # improve view selection

    vertno_max, time_max = stc.get_peak(hemi=hemi)
    stc_max = np.max(stc.data)

    if plot:
        brain_inf = stc.plot(surface='inflated', hemi=hemi, subjects_dir=subjects_dir,
                             clim=dict(kind='value', lims=[0, stc_max*0.75, stc_max]),
                             initial_time=time_max, time_unit='s', alpha=1, background='w', foreground='k')

        brain_inf.add_foci(vertno_max, coords_as_verts=True, hemi=hemi, color='blue',
                           scale_factor=0.8)

        brain_inf.add_foci(stim_coords['surf'], map_surface='pial', hemi=hemi, color='green', scale_factor=0.8)

        brain_inf.show_view(view)  # mlab.view(130, 90)
        fig_fname = op.join(study_path, 'source_stim', subj, 'figures', 'distributed', '%s_inf_foci.eps' % cond)
        brain_inf.save_image(fig_fname)
        mlab.show()
        mlab.clf()

    max_mni = mne.vertex_to_mni(vertno_max, hemis=0, subject=subj, subjects_dir=subjects_dir)  # mni coords of vertex (ojo vertex resolution ico-5)
    stim_mni = stim_coords['mni']  # mni coords of stimulation
    dist_mni = euclidean(max_mni, stim_mni)
    print(dist_mni)
    return dist_mni

    # brain_pial = stc.plot(surface='pial', hemi='lh', subjects_dir=subjects_dir,
    #                      clim=dict(kind='value', lims=[0, stc_max*0.75, stc_max]),
    #                      initial_time=time_max, time_unit='s', alpha=1, background='w', foreground='k')
    #
    # brain_pial.add_foci(vertno_max, coords_as_verts=True, hemi='lh', color='blue',
    #                    scale_factor=1)
    #
    # brain_pial.add_foci(stim_coords['surf'], hemi='lh', color='green', scale_factor=1)
    #
    # brain_pial.show_view('medial')  # mlab.view(130, 90)
    # fig_fname = op.join(study_path, 'source_stim', subj, 'figures', 'distributed', '%s_pial_foci.eps' % cond)
    # brain_pial.save_image(fig_fname)
    # mlab.show()


def dipole_source_loc(evoked, cov,  subj, study_path):
    evo_crop = evoked.copy().crop(-0.003, 0.003)
    trans_fname = op.join(study_path, 'source_stim', subj, 'source_files', '%s-trans.fif' % subj)
    bem_fname = op.join(study_path, 'freesurfer_subjects', subj, 'bem', '%s-bem-sol.fif' % subj)

    cond = eeg_epo.info['description']
    stim_coords = find_stim_coords(cond, subj, study_path)

    # Fit a dipole
    dip = mne.fit_dipole(evo_crop, cov, bem_fname, trans_fname)[0]

    import mne.transforms as tra
    # from scipy import linalg
    trans = mne.read_trans(trans_fname)
    # surf_to_head = linalg.inv(trans['trans'])

    # stim_point = tra.apply_trans(surf_to_head, stim_coords['surf']/1e3)
    # dip.pos[np.argmax(dip.gof)] = tra.apply_trans(surf_to_head, stim_coords['surf']/1e3)  # check stim loc (apply affine in m)
    stim_point = stim_coords['surf'] # get point for plot in mm

    dip_surf = tra.apply_trans(trans['trans'], dip.pos[np.argmax(dip.gof)]) * 1e3  # transform from head to surface
    dist_surf = euclidean(dip_surf, stim_coords['surf']) # compute distance in surface space

    # Plot the result in 3D brain with the MRI image.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    dip.plot_locations(trans_fname, subj, subjects_dir, mode='orthoview', coord_frame='mri', ax=ax, show_all=False)
    ax.scatter(stim_point[0], stim_point[1], stim_point[2])
    ax.plot([stim_point[0], -128], [stim_point[1], stim_point[1]], [stim_point[2], stim_point[2]], color='g')
    ax.plot([stim_point[0], stim_point[0]], [stim_point[1], -128], [stim_point[2], stim_point[2]], color='g')
    ax.plot([stim_point[0], stim_point[0]], [stim_point[1], stim_point[1]], [stim_point[2], -128], color='g')
    ax.text2D(0.05, 0.90, 'distance: %i mm \nstim coords = %0.1f %0.1f %0.1f' % (dist_surf, stim_point[0], stim_point[1], stim_point[2]),
              transform=ax.transAxes)
    red_patch = mpatches.Patch(color='red')
    green_patch = mpatches.Patch(color='green')
    fig.legend(handles=[red_patch, green_patch], labels=['dipole', 'electrode'])
    fig.savefig(op.join(study_path, 'source_stim', subj, 'figures', 'dipole', '%s_dip.png' % cond))
    print(dist_surf)
    return dist_surf




if __name__ == '__main__':
    subj = 'S5'
    study_path = '/home/eze/intra_extra'
    subjects_dir = '/home/eze/intra_extra/freesurfer_subjects'

    epo_path = op.join(study_path, 'source_stim', subj, 'epochs', 'fif')
    conds = glob.glob(epo_path + '/*-epo.fif')

    fwd_fname = op.join(study_path, 'source_stim', subj, 'source_files', '%s-fwd.fif' % subj)
    if not op.isfile(fwd_fname):
        fwd = make_fwd_solution(conds[0], subj, study_path)
    else:
        fwd = mne.read_forward_solution(fwd_fname)

    for cond_fname in conds:
        eeg_epo = mne.read_epochs(cond_fname)
        eeg_epo.interpolate_bads()
        cov = mne.compute_covariance(eeg_epo, method='shrunk', tmin=-0.5, tmax=-0.3)  # use method='auto' for final computation
        evoked = eeg_epo.average()
        # mne.viz.plot_evoked_topo(evoked)

        distr_mni_err = source_loc(evoked, cov, fwd, subj, study_path)
        dip_surf_err = dipole_source_loc(cond_fname, subj, study_path)

        cond = eeg_epo.info['description']
        stim_params = get_stim_params(cond)


        a={'subj': subj, 'w_s': stim_params['w_s'], 'intens': stim_params['stim_int'], 'ch': stim_params['ch'],
         'is_left': stim_params['is_left'], }


