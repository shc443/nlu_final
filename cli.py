from pl_classifier import NLU_dataset, NLU_DataModule, NLU_Toxic_classifier
import pickle
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
from pl_bolts.callbacks import PrintTableMetricsCallback
from sklearn.preprocessing import MultiLabelBinarizer

import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

import seaborn as sns
import matplotlib.pyplot as plt
import re
import pandas as pd

import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AdamW, BertConfig
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
from transformers import RobertaConfig, RobertaModel
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

import pandas as pd
import numpy as np
import random
import time
import datetime

#parameters
N_EPOCHS = 3
BATCH_SIZE = 12
MAX_LEN = 256
LR = 2e-05
opt_thresh = 0.4

#directory
input_path = "/scratch/shc443/nlu_final/Data/SentimentAnalysisData/Detox_Dataset"
output_path = "/scratch/shc443/nlu_final/results/analysis_results_toxic"

data = pd.read_csv(input_path)

mlb = MultiLabelBinarizer()
data['Label'] = mlb.fit_transform(data.Label.astype(str)).tolist()

steps_per_epoch= data.shape[0] // BATCH_SIZE
total_training_steps = steps_per_epoch * N_EPOCHS
warmup_steps = total_training_steps // 5
warmup_steps, total_training_steps

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
special_tokens_dict = {'additional_special_tokens': ['[TOXIC]']}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

#9:1 split
train_df, val_df = train_test_split(data, test_size=0.2)
data_module = NLU_DataModule(
  train_df,
  val_df,
  tokenizer,
  batch_size=BATCH_SIZE,
  max_token_len=MAX_LEN)
data_module.setup()

#Creating Model 
new_model = NLU_Toxic_classifier(n_classes=2,
                              n_warmup_steps=warmup_steps,
                              n_training_steps=total_training_steps)

trainer = pl.Trainer(max_epochs=N_EPOCHS, gpus=4, progress_bar_refresh_rate=3)

trainer.fit(new_model, data_module)

trainer.save_checkpoint(output_path + "DeBERTa_220417_1.ckpt")

#######################
new_model = NLU_Toxic_classifier.load_from_checkpoint(
            checkpoint_path=output_path+"DeBERTa_220417_1.ckpt",
            n_classes=2)

data_module = NLU_DataModule(
  data,
  data,
  tokenizer,
  batch_size=BATCH_SIZE,
  max_token_len=MAX_LEN)

data_module.setup()

testing_predict = trainer.predict(new_model, datamodule=data_module)

