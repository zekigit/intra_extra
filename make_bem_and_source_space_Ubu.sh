#!/usr/bin/env bash
# use as -> source make_bem_and_source_space.sh subj_id

# Freesurfer, MNE-C, Python-MNE and subjects dir paths
#export FREESURFER_HOME=/usr/local/freesurfer
#export SUBJECTS_DIR=/home/eze/fs_subjects
source $FREESURFER_HOME/SetUpFreeSurfer.sh

export SUBJECT=$1
export STUDY_PATH=/home/eze/intra_extra

export MNE_ROOT=/home/eze/soft/MNE-2.7.4-3514-Linux-x86_64
. $MNE_ROOT/bin/mne_setup_sh


echo "Pocessing subject: " ${SUBJECT}


# Run BEM solution
#python /home/eze/PycharmProjects/intra_extra_ubu/make_bem.py ${SUBJECT} ${SUBJECTS_DIR}

# Run space sources
python /home/eze/PycharmProjects/intra_extra_ubu/make_source_space.py ${SUBJECT} ${SUBJECTS_DIR} ${STUDY_PATH}

