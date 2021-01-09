"""
Reconstruct fully sampled OCMR datasets

"""
# Read the OCMR dataset
import numpy as np
import matplotlib.pyplot as plt
import math

from ismrmrdtools import show, transform
# import ReadWrapper
import read_ocmr as read

# Load the data, display size of kData and scan parmaters
filename = './ocmr_data/fs_0001_1_5T.h5'
kData, param = read.read_ocmr(filename);
print('Dimension of kData: ', kData.shape)

# %% Inverse FFT, Sum of Square (SoS) coil combination, Remove RO oversampling
# Image reconstruction (SoS)
dim_kData = kData.shape;
CH = dim_kData[3];
SLC = dim_kData[6];
kData_tmp = np.mean(kData, axis=8);  # average the k-space if average > 1

im_coil = transform.transform_kspace_to_image(kData_tmp, [0, 1]);  # IFFT (2D image)
im_sos = np.sqrt(np.sum(np.abs(im_coil) ** 2, 3));  # Sum of Square
print('Dimension of Image (with ReadOut ovesampling): ', im_sos.shape)
RO = im_sos.shape[0];
image = im_sos[math.floor(RO / 4):math.floor(RO / 4 * 3), :, :];  # Remove RO oversampling
print('Dimension of Image (without ReadOout ovesampling): ', image.shape)

# %% Display the reconstructed cine image (central slice)
# Show the reconstructed cine image
from IPython.display import clear_output
import time

slc_idx = math.floor(SLC / 2);
print(slc_idx)
image_slc = np.squeeze(image[:, :, :, :, :, :, slc_idx]);
for rep in range(5):  # repeate the movie for 5 times
    for frame in range(image_slc.shape[2]):
        clear_output(wait=True)
        plt.imshow(image_slc[:, :, frame], vmin=0, vmax=0.6 * np.amax(image_slc), cmap='gray');
        plt.axis('off');
        plt.show()
        time.sleep(0.03)
