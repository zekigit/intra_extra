import nibabel as nb

mri_file = '/Users/lpen/Documents/intra_milano/hbp_jupyter/MRI.img'
img = nb.load(mri_file)
nb.save(img, '/Users/lpen/Documents/intra_milano/hbp_jupyter/MRI.nii')


mri_file = '/Users/lpen/Documents/intra_milano/hbp_jupyter/hbp_subj/mri/T1.mgz'
img = nb.load(mri_file)
