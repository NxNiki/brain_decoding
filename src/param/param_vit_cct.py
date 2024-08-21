import numpy as np 
import os


param_dict={
    'lr': 1e-04,
    'batch_size': 128,
    'weight_decay': 1e-4,
    'epochs': 100,
    'lr_drop': 50,
    'validation_step': 25,
    'num_labels': 8,
    'merge_label': True,
    'hidden_size': 192,
    'num_hidden_layers': 4,
    'num_attention_heads': 6,
    'patch_size': (1, 5),
    # 'intermediate_size': 192 * 2,
    # path
    # 'movie_label_path': '/mnt/SSD2/yyding/Datasets/12concepts/12concepts_john.npy',
    'movie_label_path': '/mnt/SSD2/yyding/Datasets/12concepts/8concepts_merged.npy',
    'spike_path': '/mnt/SSD2/yyding/Datasets/neuron/spike_data',
    'lfp_path': '/mnt/SSD2/yyding/Datasets/neuron/lfp_data',
    'lfp_data_mode': 'sf2000-bipolar-region-clean',
    'spike_data_mode': 'notch CAR'
    # 'spike_data_mode': 'overlap'
}