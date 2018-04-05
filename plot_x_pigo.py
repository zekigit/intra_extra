from pyface.qt import QtGui, QtCore
import mne
import numpy as np
import os.path as op
from intra_extra_fx import loadmat
from surfer import Brain
import mayavi.mlab as mlab
import matplotlib.pyplot as plt

subjects_dir = '/home/eze/fs_subjects'
folder_path = '/home/eze/intra_extra/plot_x_pigo'
fname = 'sources.mat'

base_data = loadmat(op.join(folder_path, fname))

subj = 'NOBILI'

dat = base_data['SSPII_Currents']['surrogates']['statistics']['Norm']
dat_lh = dat[:2562]
dat_rh = dat[2562:]

rr, tris = mne.read_surface('/home/eze/fs_subjects/NOBILI/surf/lh.pial')

plt.hist(rr)
plt.legend(['x', 'y', 'z'])

verts = base_data['SSPII_Currents']['Mesh']['vert']*10
trian = base_data['SSPII_Currents']['Mesh']['face']


fig = mlab.figure()
mlab.triangular_mesh(verts[:,0], verts[:,1], verts[:,2], trian)




ex_dat = np.random.randn(len(rr), 2)
br = Brain(subj, hemi='lh', surf='pial', subjects_dir=subjects_dir)
br.add_data(ex_dat, hemi='lh')



