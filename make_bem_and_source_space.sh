#!/usr/bin/env bash
# use as -> source make_bem_and_source_space.sh subj_id

# Freesurfer, MNE-C, Python-MNE and subjects dir paths
export FREESURFER_HOME=/Applications/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh

export SUBJECTS_DIR=/Volumes/MAXTOR/freesurfer_subjects
export SUBJECT=$1
export MNE_ROOT=/Applications/MNE-2.7.4-3378-MacOSX-x86_64

source $FREESURFER_HOME/SetUpFreeSurfer.sh
. $MNE_ROOT/bin/mne_setup_sh


echo "Pocessing subject: " ${SUBJECT}

# Run BEM solution
python3 /Users/lpen/PycharmProjects/intra_extra/make_bem.py ${SUBJECT} ${SUBJECTS_DIR}

# Run space sources
source activate py2mne
python /Users/lpen/PycharmProjects/intra_extra/make_source_space.py ${SUBJECT}

# Run coregistration
#mne coreg