import mne
import os.path as op
from os import listdir
import pandas as pd
import numpy as np
from intra_extra_info import study_path
from intra_extra_fx import load_eeg, find_stim_events, make_dig_montage_file
pd.set_option('display.expand_frame_repr', False)
import sys


def preprocess(fname_dat, subj, study_path, bads):
    cond = op.split(fname_dat)[-1]
    raw = load_eeg(fname_dat, subj, study_path)
    if bads == 'load':
        all_bads = pd.read_csv(op.join(study_path, 'source_stim', subj, 'epochs', '%s_bad_chans.csv' % subj))
        bads_bool = all_bads[all_bads.cond == cond].values[0][2:]
        bads = [raw.ch_names[ix] for ix, bo in enumerate(bads_bool) if bo == 1]

    raw.info['bads'] = bads
    #raw.filter(0.1, None)

    raw_ok = False
    while not raw_ok:
        raw.plot(n_channels=64, block=True, title=cond, duration=15)  # mark bad channels
        events, raw_ok = find_stim_events(raw)

    eeg_epo = mne.Epochs(raw, events, event_id={'stim': 1}, tmin=-0.5, tmax=0.5, baseline=(-0.5, -0.25),
                         reject_by_annotation=False, preload=True)
    dig_fname = op.join(study_path, 'physio_data', subj, 'chan_info', '%s_egi_digitalization.hpts' % subj)
    montage = mne.channels.read_montage(dig_fname, unit='mm')
    # montage.plot(kind='3d')
    eeg_epo.set_montage(montage, set_dig=True)
    eeg_epo.info['description'] = cond

    eeg_epo.set_eeg_reference('average', projection=True)
    eeg_epo.apply_proj()
    eeg_epo.plot(n_channels=64, n_epochs=len(eeg_epo), scalings={'eeg': 20e-5}, block=True, title=cond)

    epo_fname = op.join(study_path, 'source_stim', subj, 'epochs', 'fif', '%s-epo.fif' % eeg_epo.info['description'])
    eeg_epo.save(epo_fname)
    print('saving')
    return eeg_epo, eeg_epo.info['bads']


if __name__ == '__main__':
    subj = sys.argv[1]
    # study_path = sys.argv[2]

    # subj = 'S3'
    dig_fname = op.join(study_path, 'physio_data', subj, 'chan_info', '%s_egi_digitalization.hpts' % subj)
    if not op.isfile(dig_fname):
        make_dig_montage_file(subj, study_path)

    dat_path = op.join(study_path, 'physio_data', subj, 'SPES')
    stim_files = listdir(dat_path)

    # stim_file = stim_files[21]
    bads = []
    for stim_file in stim_files:
        fname_dat = op.join(dat_path, stim_file)
        eeg_epo, bads = preprocess(fname_dat, subj, study_path, bads)
