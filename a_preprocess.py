import mne
import glob
import os.path as op
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('qt4agg')
import matplotlib.pyplot as plt
from intra_extra_info import study_path
from intra_extra_fx import make_dig_montage_file
import sys
pd.set_option('display.expand_frame_repr', False)

# matplotlib.interactive(True)


def load_data(subj, cond_fname):
    eeg_raw = mne.io.read_raw_edf(cond_fname, preload=True, stim_channel=-1)
    #eeg_raw._data[:-1, :] /= 1e6  # scale to volts

    #eeg_raw.plot(n_channels=32, scalings={'eeg': 2e-4})
    events = mne.find_events(eeg_raw)
    events[:, 0] -= 2
    eeg_epo = mne.Epochs(eeg_raw, events, event_id={'stim': 1}, tmin=-0.5, tmax=0.5, baseline=(-0.5, -0.3), preload=True)
    dig_fname = op.join(study_path, 'physio_data', subj, 'chan_info', '%s_egi_digitalization.hpts' % subj)
    montage = mne.channels.read_montage(dig_fname, unit='mm')
    # montage.plot(kind='3d')

    eeg_epo.set_montage(montage, set_dig=True)
    eeg_epo.info['description'] = op.split(cond_fname)[-1].replace('_epochs.edf', '')
    #eeg_epo.plot_sensors(kind='3d')
    return eeg_epo


def preprocess(eeg_epo):
    # bads = bad_chans[subj]
    # eeg_epo.info['bads'] = bads
    # eeg_epo = eeg_epo[0]
    eeg_epo.filter(0.1, None, method='iir', iir_params=None)
    eeg_epo.plot(n_channels=64, scalings={'eeg': 4e-4}, n_epochs=len(eeg_epo), block=True)
    # plt.clf()
    # plt.close()
    eeg_epo.set_eeg_reference('average', projection=True)
    eeg_epo.apply_proj()
    epo_fname = op.join(study_path, 'source_stim', subj, 'epochs', 'fif', '%s-epo.fif' % eeg_epo.info['description'])
    eeg_epo.save(epo_fname)
    print('saving')
    return eeg_epo


if __name__ == '__main__':
    # subj = sys.argv[1]
    # study_path = sys.argv[2]

    subj = 'S3'
    dig_fname = op.join(study_path, 'physio_data', subj, 'chan_info', '%s_egi_digitalization.hpts' % subj)
    if not op.isfile(dig_fname):
        make_dig_montage_file(subj, study_path)

    epo_path = op.join(study_path, 'source_stim', subj, 'epochs', 'edf')
    conds = glob.glob(epo_path + '/*.edf')

    for cond_fname in conds[-3]:
        eeg_epo = load_data(subj, cond_fname)
        eeg_epo = preprocess(eeg_epo)
