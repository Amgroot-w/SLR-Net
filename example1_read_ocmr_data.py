"""
An example of reading k-space from OCMR data

"""
# %% import required library
import numpy as np
import matplotlib.pyplot as plt
import math
from ismrmrdtools import show, transform
import read_ocmr as read  # import ReadWrapper
import pprint

# %% Read k-space from h5 file and display scan parameters
# Load the data, display size of kData and scan parmaters
filename = './ocmr_data/fs_0001_1_5T.h5'
kData, param = read.read_ocmr(filename)
print('Dimension of kData: ', kData.shape)
print('Scan paramters:')
pprint.pprint(param)

# %% Display the sampling pattern
# Show the sampling Pattern
# kData_tmp-[kx,ky,kz,coil,phase,set,slice,rep], samp-[kx,ky,kz,phase,set,slice,rep]
dim_kData = kData.shape
CH = dim_kData[3]
SLC = dim_kData[6]
kData_tmp = np.mean(kData, axis=8)  # average the k-space if average > 1
samp = (abs(np.mean(kData_tmp, axis=3)) > 0).astype(np.int)  # kx ky kz phase set slice

slc_idx = math.floor(SLC / 2)
fig1 = plt.figure(1)
fig1.suptitle("Sampling Pattern", fontsize=14)
plt.subplot2grid((1, 8), (0, 0), colspan=6)
tmp = plt.imshow(np.transpose(np.squeeze(samp[:, :, 0, 0, 0, slc_idx])), aspect='auto')
plt.xlabel('kx')
plt.ylabel('ky')
tmp.set_clim(0.0, 1.0)  # ky by kx
plt.subplot2grid((1, 9), (0, 7), colspan=2)
tmp = plt.imshow(np.squeeze(samp[int(dim_kData[0] / 2), :, 0, :, 0, slc_idx]), aspect='auto')
plt.xlabel('frame')
plt.yticks([])
tmp.set_clim(0.0, 1.0)  # ky by frame
plt.show()

# %% Display the time averaged image
# Average the k-sapce along phase(time) dimension
kData_sl = kData_tmp[:, :, :, :, :, :, slc_idx, 0];
samp_avg = np.repeat(np.sum(samp[:, :, :, :, :, slc_idx, 0], 3), CH, axis=3) + np.finfo(float).eps
kData_sl_avg = np.divide(np.squeeze(np.sum(kData_sl, 4)), np.squeeze(samp_avg));

im_avg = transform.transform_kspace_to_image(kData_sl_avg, [0, 1]);  # IFFT (2D image)
im = np.sqrt(np.sum(np.abs(im_avg) ** 2, 2))  # Sum of Square
fig2 = plt.figure(1);
plt.imshow(np.transpose(im), vmin=0, vmax=0.8 * np.amax(im), cmap='gray');
plt.axis('off');  # Show the image
plt.show()

