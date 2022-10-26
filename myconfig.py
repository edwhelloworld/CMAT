# -*- coding: UTF-8 -*-
import numpy as np

train_batch_size = 1
eval_batch_size = 1
learning_rate = 5e-5
cnn_lr=1e-4
mi_lr=1e-3
num_train_epochs = 70.0
data_dir = 'data/twitter2017'
task_name = 'twitter2017'#remember 2 change both
bert_model = 'bert-base-cased'

mm_model = 'MTCCMBert'
fine_tune_cnn = True

contrastive_margin = 0.1
contrastive_ratio = 2
mi_ratio = 1

testData_asDev = True
output_dir = './outputs/'
cache_dir = "./"
max_seq_length = 128
do_train = True
do_eval = True
do_lower_case = False
warmup_proportion = 0.1
no_cuda = False
local_rank = -1
seed = 32
gradient_accumulation_steps = 1 #Number of updates steps to accumulate before performing a backward/update pass
fp16 = False
loss_scale = 0
layer_num1 = 1
layer_num2 = 1
layer_num3 = 1

resnet_root = './resnet'
crop_size = 224
path_image = '../twit_subimages/'
server_ip = ''
server_port = ''

