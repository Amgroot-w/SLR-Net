
# %%
import tensorflow as tf
import os
from model import S_Net, SLR_Net
from dataset_tfrecord import get_dataset
import argparse
import scipy.io as scio
import mat73
import numpy as np
from datetime import datetime
import time
from tools.tools import video_summary
from tools.tools import tempfft, mse, loss_function_ISTA

import read_ocmr as read  # import ReadWrapper

parser = argparse.ArgumentParser()
parser.add_argument('--num_epoch', metavar='int', nargs=1, default=['50'], help='number of epochs')
parser.add_argument('--batch_size', metavar='int', nargs=1, default=['1'], help='batch size')
parser.add_argument('--learning_rate', metavar='float', nargs=1, default=['0.001'], help='initial learning rate')
parser.add_argument('--niter', metavar='int', nargs=1, default=['10'], help='number of network iterations')
parser.add_argument('--acc', metavar='int', nargs=1, default=['12'], help='accelerate rate')
parser.add_argument('--net', metavar='str', nargs=1, default=['SLRNET'], help='SLR Net or S Net')
parser.add_argument('--gpu', metavar='int', nargs=1, default=['0'], help='GPU No.')
parser.add_argument('--data', metavar='str', nargs=1, default=['DYNAMIC_V2_MULTICOIL'], help='dataset name')
parser.add_argument('--learnedSVT', metavar='bool', nargs=1, default=['True'], help='Learned SVT threshold or not')
args = parser.parse_args()

# %%  读入数据
filename = './ocmr_data/fs_0001_1_5T.h5'
kData, param = read.read_ocmr(filename)

# %%  参数、log和model的路径设置
mode = 'training'
dataset_name = args.data[0].upper()
batch_size = int(args.batch_size[0])
num_epoch = int(args.num_epoch[0])
learning_rate = float(args.learning_rate[0])

acc = int(args.acc[0])
net_name = args.net[0].upper()
niter = int(args.niter[0])
learnedSVT = bool(args.learnedSVT[0])

logdir = './logs'
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
model_id = TIMESTAMP + net_name + '_' + dataset_name + str(acc) + '_lr_' + str(learning_rate)
summary_writer = tf.summary.create_file_writer(os.path.join(logdir, mode, model_id + '/'))

modeldir = os.path.join('models/stable/', model_id)
os.makedirs(modeldir)

# prepare undersampling mask
multi_coil = True
mask_size = '18_192_192'

# %%
mask = mat73.loadmat('/data1/ziwenke/SLRNet/mask_newdata/vista_' + mask_size + '_acc_12.mat')['mask']
mask = tf.cast(tf.constant(mask), tf.complex64)







