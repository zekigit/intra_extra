import mne
import os.path as op
import pandas as pd
import numpy as np
from ieeg_fx import loadmat, make_bnw_nodes
from intra_extra_info import subj_dig_mont, study_path
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_locs(subject, study_path, condition):
    seeg_loc_file = op.join(study_path, 'physio_data', subject, 'chan_info', '%s_%s_seeg_ch_info.csv' % (subject, condition))
    eeg_loc_file = op.join(study_path, 'physio_data', subject, 'EGI_contacts.mat')

    seeg_loc = pd.read_csv(seeg_loc_file)
    seeg_loc['subject'] = subject
    eeg_loc = loadmat(eeg_loc_file)
    return seeg_loc


def create_seeg_info(seeg_loc):
    dig_ch_pos = dict(zip(seeg_loc['name'], 1e-3 * np.array(seeg_loc[['x_surf', 'y_surf', 'z_surf']])))
    montage = mne.channels.DigMontage(dig_ch_pos=dig_ch_pos)
    info = mne.create_info(seeg_loc['name'].tolist(), 1000., 'seeg', montage=montage)
    return info


def plot_locs(seeg_loc, info, study_path):
    from mne.viz import plot_alignment
    from mayavi import mlab
    subjects_dir = op.join(study_path, 'freesurfer_subjects')
    subject = seeg_loc['subject'][0]

    fig = plot_alignment(info, subject=seeg_loc['subject'][0], subjects_dir=subjects_dir,
                         surfaces=['pial'], meg=False, coord_frame='head')
    mlab.view(200, 70)
    mlab.show()

    file_nodes = op.join(study_path, 'physio_data', subject, 'chan_info', '%s.node' % subject)
    make_bnw_nodes(file_nodes, coords=seeg_loc[['x', 'y', 'z']], colors=1., sizes=1.)


def load_eeg(fname_dat, subj, study_path):
    # load EEG data
    eeg_fname = op.join(fname_dat, 'HDEEG', 'data.mat')
    try:
        eeg_base = loadmat(eeg_fname)['HDEEG']
    except KeyError:
        eeg_base = loadmat(eeg_fname)['EEG']

    ch_names = subj_dig_mont[subj]['ch_names']

    dig_fname = op.join(study_path, 'physio_data', subj, 'chan_info', '%s_egi_digitalization.hpts' % subj)
    montage = mne.channels.read_montage(dig_fname, unit='mm')

    eeg_info = mne.create_info(['E{}' .format(n+1) for n in range(256)], 1000., 'eeg', montage=montage)

    eeg_raw = mne.io.RawArray(eeg_base['data'] * 1e-6, eeg_info)  # rescale to volts
    eeg_raw.set_montage(montage, set_dig=True)
    # eeg_raw.plot_sensors(kind='3d', ch_type='all', show_names=True )
    eeg_raw.info['description'] = op.split(fname_dat)[-1].replace('_epochs.edf', '')
    # eeg_raw.notch_filter(np.arange(50, 251, 50), filter_length='auto')
    return eeg_raw


def check_seeg_chans(seeg_base, seeg_loc):
    all_rec_chans = [str(a.Name) for a in seeg_base['CM']['Channel']]

    # Get brain channels
    rec_chans = list()
    rec_ixs = list()
    for ix_ch, ch in enumerate(all_rec_chans):
        if len(ch) > 1:
            if sum(c.isalpha() for c in ch) == 1:
                if '\'' in ch:
                    match = re.match(r"([A-Z])'([0-9]+)", ch)
                    if match:
                        items = match.groups()
                        if int(items[1]) < 20:
                            rec_chans.append(ch)
                            rec_ixs.append(ix_ch)
                else:
                    match = re.match(r"([A-Z])([0-9]+)", ch)
                    if match:
                        items = match.groups()
                        if int(items[1]) < 20:
                            rec_chans.append(ch)
                            rec_ixs.append(rec_ixs)

    # Check missing inv commas
    for ix_ch, ch in enumerate(rec_chans):
        pass  # add criterion
    seeg_rec = seeg_loc[seeg_loc['name'].isin(rec_chans)]

    return rec_chans, rec_ixs, seeg_rec


def export_slicer_markups_egi(subj, study_path):
    dig_fname = op.join(study_path, 'physio_data', subj, 'EGI_contacts.mat')
    dig_points = loadmat(dig_fname)['Digitalization']['LocalizationMRI']

    dig_points[:, 0] = dig_points[:, 0]*-1
    dig_points[:, 1] = dig_points[:, 1]*-1

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(dig_points[:, 0], dig_points[:, 1], dig_points[:, 2])
    for x, y, z, lab in zip(dig_points[:, 0], dig_points[:, 1], dig_points[:, 2], np.arange(len(dig_points))):
        ax.text(x, y, z, lab)
    print(dig_points.shape)

    ch_names = subj_dig_mont[subj]['ch_names']

    header = ['# Markups fiducial file version = 4.5', '# CoordinateSystem = 0', '# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID']
    elec_info = ['vtkMRMLMarkupsFiducialNode_{},{},{},{},0,0,0,1,1,1,0,{},,' .format(ix+1, x, y, z, lab) for ix, (x, y, z, lab) in
                 enumerate(zip(dig_points[:,0], dig_points[:,1], dig_points[:,2], ch_names))]

    dat_to_write = header + elec_info
    with open(op.join(study_path, 'physio_data', subj, 'chan_info', 'egi_locs_ori.fcsv'), 'w') as fid:
        fid.writelines('%s\n' % l for l in dat_to_write)


def make_dig_montage_file(subj, study_path):
    # load digitization
    dig_fname = op.join(study_path, 'physio_data', subj, 'EGI_contacts.mat')
    dig_points = loadmat(dig_fname)['Digitalization']['LocalizationMRI']

    dig_points[:, 0] = dig_points[:, 0]*-1
    dig_points[:, 1] = dig_points[:, 1]*-1

    ch_names = subj_dig_mont[subj]['ch_names']
    ch_types = subj_dig_mont[subj]['ch_types']
    dig_sel = subj_dig_mont[subj]['dig_sel']
    dig_points = dig_points[dig_sel]

    # ch_names = np.array(['1', '3'] + ['E' + str(i + 1) for i in range(256)] + ['2'])
    # ch_types = np.array([['fid'] + ['fid'] + ['eeg']*(len(ch_names)-3) + ['fid']])

    hpts_fname = op.join(study_path, 'physio_data', subj, 'chan_info', '%s_egi_digitalization.hpts' % subj)

    hpts = np.zeros(len(ch_names), dtype=[('ch_type', 'S6'), ('ch_name', 'S6'), ('x', float), ('y', float), ('z', float)])
    hpts['ch_type'] = ch_types
    hpts['ch_name'] = ch_names
    hpts['x'] = dig_points[:, 0]
    hpts['y'] = dig_points[:, 1]
    hpts['z'] = dig_points[:, 2]

    np.savetxt(hpts_fname, hpts, fmt="%3s %3s %10.3f %10.3f %10.3f")


def find_stim_coords(cond, subj, study_path):
    seeg_ch_info = pd.read_csv(op.join(study_path, 'physio_data', subj, 'chan_info', '%s_seeg_ch_info.csv' % subj))

    is_left = cond.find('\'') != -1
    if is_left:
        match = re.search(r"[A-Z]'[0-9]+", cond)
    else:
        match = re.search(r"[A-Z][0-9]+", cond)
    if match:
        ch = match.group()

    ch1_coords = {'surf': seeg_ch_info[['x_surf', 'y_surf', 'z_surf']].loc[seeg_ch_info['name'] == ch],
                  'scanner_RAS': seeg_ch_info[['x_mri', 'y_mri', 'z_mri']].loc[seeg_ch_info['name'] == ch],
                  'mni': seeg_ch_info[['x_mni152', 'y_mni152', 'z_mni152']].loc[seeg_ch_info['name'] == ch]}

    ch2 = '{}{}' .format(ch[0] if not is_left else ch[:2], int(re.findall(r'[1-9]+', ch)[0])+1)

    ch2_coords = {'surf': seeg_ch_info[['x_surf', 'y_surf', 'z_surf']].loc[seeg_ch_info['name'] == ch2],
                  'scanner_RAS': seeg_ch_info[['x_mri', 'y_mri', 'z_mri']].loc[seeg_ch_info['name'] == ch2],
                  'mni': seeg_ch_info[['x_mni152', 'y_mni152', 'z_mni152']].loc[seeg_ch_info['name'] == ch2]}

    stim_coords = {'surf': np.average([ch1_coords['surf'].values, ch2_coords['surf'].values], axis=0)[0],
                   'scanner_RAS': np.average([ch1_coords['scanner_RAS'].values, ch2_coords['scanner_RAS'].values], axis=0)[0],
                   'mni': np.average([ch1_coords['mni'].values, ch2_coords['mni'].values], axis=0)[0]}


    surf_ori_to_sli_trans_fname = op.join(study_path, 'source_stim', subj, 'source_files',
                                          'surf_ori_to_surf_sli_trans.tfm')

    if op.isfile(surf_ori_to_sli_trans_fname):
        surf_ori_to_sli_trans = np.loadtxt(surf_ori_to_sli_trans_fname)
        from scipy.linalg import inv
        from mne.transforms import apply_trans as at

        sli_to_ori = inv(surf_ori_to_sli_trans)
        coords_surf_ori = at(sli_to_ori, stim_coords['scanner_RAS'])

        stim_coords['surf_ori'] = coords_surf_ori.copy()
    return stim_coords


def get_stim_params(cond):
    spl_cond = cond.split('_')
    stim_ch = spl_cond[0]
    w_s = spl_cond[1]
    stim_int = spl_cond[2]
    is_left = stim_ch.find('\'') != -1
    stim_params = {'ch': stim_ch, 'is_left': is_left, 'w_s': w_s, 'stim_int': stim_int}
    return stim_params


def plot_source_space(subj, study_path, subjects_dir):
    from surfer import Brain  # noqa
    import mayavi.mlab as mlab
    src_fname = op.join(study_path, 'source_stim', subj, 'source_files', '%s-oct5-src.fif' % subj)
    src = mne.read_source_spaces(src_fname, patch_stats=True)

    brain = Brain(subj, 'lh', 'inflated', subjects_dir=subjects_dir)
    surf = brain.geo['lh']

    vertidx = np.where(src[0]['inuse'])[0]

    mlab.points3d(surf.x[vertidx[:-1]], surf.y[vertidx[:-1]],
                  surf.z[vertidx[:-1]], color=(1, 1, 0), scale_factor=1.5)
    mlab.show()


def save_results(subj, study_path, stim_params, evoked, img_type, dist_dis, dist_dip):
    import csv
    cols = ['subj', 'w_s', 'intens', 'ch', 'is_left', 'n_epo', 'n_ch', 'dist_dis', 'dist_dip']
    results_fname = op.join(study_path, 'source_stim', subj, 'results', '%s_%s_results.csv' % (subj, img_type))
    if not op.isfile(results_fname):
        with open(results_fname, 'w') as fid:
            wr = csv.writer(fid, delimiter=',', quotechar='\'')
            wr.writerow(cols)

    if dist_dip is None:
        dist_dip = 999
    if dist_dis is None:
        dist_dip = 999

    results = [subj, stim_params['w_s'], stim_params['stim_int'], stim_params['ch'], stim_params['is_left'], evoked.nave,
               len(evoked.ch_names), round(dist_dis, 2), round(dist_dip, 2)]
    with open(results_fname, 'a') as fid:
        wr = csv.writer(fid, delimiter=',', quotechar='\'')
        wr.writerow(results)

def read_an_to_ori_trans(an_to_ori_fname):
    an_to_ori_trans = np.genfromtxt(an_to_ori_fname, skip_header=8, skip_footer=18)
    return an_to_ori_trans


def find_stim_events(raw):
    import peakutils
    good_ch = [ix for ix, ch in enumerate(raw.ch_names) if ch not in raw.info['bads']]
    dat = raw.get_data(picks=good_ch)

    hp = mne.filter.filter_data(dat, raw.info['sfreq'], 200, None, method='iir', iir_params=None)
    avg = hp.mean(0)
    gfp = hp.std(0)

    use = gfp

    ev_ok = False
    raw_ok = True
    thres = 0.7
    while not ev_ok:
        indexes = peakutils.indexes(use, thres=thres, min_dist=100)
        plt.plot(raw.times, use)
        plt.plot(indexes/1e3, use[indexes], 'o')
        plt.title('threshold = %0.1f - stimulations found: %s' % (thres, len(indexes))), plt.ylabel('gfp'), plt.xlabel('time (s)')
        plt.tight_layout()
        plt.show()
        resp = raw_input('ok? y (yes) / r (back to raw) / nr (new threshold): ')
        if resp == 'y':
            ev_ok = True
        elif resp == 'r':
            ev_ok = True
            raw_ok = False
        else:
            thres = float(resp)
    events = np.vstack((indexes, np.zeros(len(indexes)), np.ones(len(indexes)))).T.astype(int)

    return events, raw_ok


def make_static_trans(subj):
    from mne.transforms import Transform
    fro=4
    to=5
    trans_mat = np.array([[1., 0., 0., 0.],
                          [0., 1., 0., 0.],
                          [0., 0., 1., 0.],
                          [0., 0., 0., 1.]])
    trans = Transform(fro, to, trans_mat)
    fname_trans = op.join(study_path, 'source_stim', subj, 'source_files', 'orig', '%s_static-trans.fif' % subj)
    trans.save(fname_trans)