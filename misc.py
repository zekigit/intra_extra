import numpy as np
import mne
import matplotlib.pyplot as plt

y1 = np.hstack((np.repeat(0, 499), 0., 1., -1., 0,  np.repeat(0, 498)))
y2 = np.hstack((np.repeat(0, 499), 0, 0.25, -0.25,0, np.repeat(0, 498)))
noise = np.random.normal(0, 0.01, len(y1))
y1 += noise
y2 += noise


x = np.arange(0, len(y1), 1)

y1f = mne.filter.filter_data(y1, 1000, None, 200)
y2f = mne.filter.filter_data(y2, 1000, None, 200)

fig, axes = plt.subplots(1, 2)
for ix, (raw, filt) in enumerate(zip([y1, y2], [y1f, y2f])):
    axes[ix].plot(x, raw)
    axes[ix].plot(x, filt)
    axes[ix].set_xlim([480, 520])
    axes[ix].set_ylim([-1, 1])
