import mne
import numpy as np
import os.path as op
import pandas as pd
from intra_extra_info import study_path
from intra_extra_fx import load_locs, create_seeg_info, plot_locs

pd.set_option('display.expand_frame_repr', False)

subject = 'S1'
condition = 'SPONT'

seeg_loc = load_locs(subject, study_path, condition)
seeg_info = create_seeg_info(seeg_loc)
plot_locs(seeg_loc, seeg_info, study_path)



