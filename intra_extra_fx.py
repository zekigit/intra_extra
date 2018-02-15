import mne
import os.path as op
import pandas as pd
import numpy as np
from ieeg_fx import loadmat, make_bnw_nodes
import re


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


def load_eeg(cond_fpath, subj):
    # load EEG data
    eeg_fname = op.join(cond_fpath, 'HDEEG', 'data.mat')
    try:
        eeg_base = loadmat(eeg_fname)['HDEEG']
    except KeyError:
        eeg_base = loadmat(eeg_fname)['EEG']

    ch_names = ['lpa', 'rpa'] + ['E' + str(i + 1) for i in range(256)] + ['nasion']

    # load digitization
    dig_fname = op.join(cond_fpath.split(subj)[0], subj, 'EGI_contacts.mat')
    dig_points = loadmat(dig_fname)['Digitalization']['LocalizationMRI']

    dig_montage = mne.channels.read_dig_montage(hsp=dig_points, elp=dig_points, point_names=ch_names, unit='mm', transform=False)

    montage = mne.channels.read_montage('EGI_256')
    eeg_info = mne.create_info(['E' + str(i+1) for i in range(256)], 1000., 'eeg', montage=montage)

    eeg_raw = mne.io.RawArray(eeg_base['data'] * 1e-6, eeg_info)  # rescale to volts
    eeg_raw.set_montage(dig_montage)

    return eeg_raw


def load_phys(subject, study_path):
    # Load SEEG
    cond_fpath = op.join(study_path, 'physio_data', subject, 'BLINK', 'SPONT')
    seeg_fname = op.join(cond_fpath, 'SEEG', 'data.mat')
    seeg_base = loadmat(seeg_fname)['SEEG']
    seeg_loc = load_locs(subject, study_path)

    dig_ch_pos = dict(zip(seeg_loc['name'], 1e-3 * np.array(seeg_loc[['x', 'y', 'z']])))
    montage = mne.channels.DigMontage(dig_ch_pos=dig_ch_pos)
    seeg_info = mne.create_info(seeg_loc['name'].tolist(), 1000., 'seeg', montage=montage)
    seeg_raw = mne.io.RawArray(seeg_base['data'], seeg_info)

    # Load HDEEG
    eeg_fname = op.join(cond_fpath, 'HDEEG', 'data.mat')
    eeg_base = loadmat(eeg_fname)['HDEEG']
    montage = mne.channels.read_montage('EGI_256')
    eeg_info = mne.create_info(['E' + str(i+1) for i in range(256)], 1000., 'eeg', montage=montage)
    eeg_raw = mne.io.RawArray(eeg_base['data'] * 1e-6, eeg_info)  # rescale to volts
    return seeg_base, eeg_raw


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


