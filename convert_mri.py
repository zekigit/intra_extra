import nibabel as nb
import os.path as op

study_path = '/home/eze/intra_extra/varios/'
subj='S3'

mri_file = op.join(study_path, subj, 'MRI.img')
img = nb.load(mri_file)

nb.save(img, op.join(study_path, subj, 'MRI_an.mgz'))


mri_file = '/Users/lpen/Documents/intra_milano/hbp_jupyter/hbp_subj/mri/T1.mgz'
img = nb.load(mri_file)
om.dicomwrappers.wrapper_from_file(fname)