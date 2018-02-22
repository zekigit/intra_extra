#!/usr/bin/env bash

export FREESURFER_HOME=/Applications/freesurfer
export SUBJECTS_DIR=/Volumes/MAXTOR/freesurfer_subjects
export SUBJECT=$1
export MNE_ROOT=/Applications/MNE-2.7.4-3378-MacOSX-x86_64
export STUDY_PATH=/Volumes/MAXTOR

source $FREESURFER_HOME/SetUpFreeSurfer.sh
. $MNE_ROOT/bin/mne_setup_sh

mne make_scalp_surfaces -s ${SUBJECT} -f -o