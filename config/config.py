import torch

# Define hyper parameter
use_cuda = True if torch.cuda.is_available() else False

# for dataset
atc_path = 'dataset/specific_atc_list.xlsx'
pro_path = 'dataset/specific_pro_list.xlsx'
lab_path = 'dataset/lab_list.xlsx'
diag_path = 'dataset/icd10_list.xlsx'
data_path = 'dataset/dataset.xlsx'
# for preprocess
isSplit = True
# for training
num_epochs = 80
batch_size = 16
learning_rate = 1e-3
workers = 8
# for model
encoder_embedding_size = 64
encoder_output_size = 64
decoder_hidden_size = encoder_output_size
teacher_forcing_ratio = .5
# max_length = 20

# for logging
checkpoint_name = 'auto_encoder.pt'
