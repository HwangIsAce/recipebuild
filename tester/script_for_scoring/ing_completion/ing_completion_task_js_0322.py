import torch
import random
import torch.backends.cudnn as cudnn
import numpy as np
##### seed 고정하기
import torch
import random
import torch.backends.cudnn as cudnn
import numpy as np

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(0)

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from itertools import chain
from typing import Optional

## custom load
import bootstrap
import datasets
import evaluate
# custom config load
import torch
import transformers
from datasets import load_dataset
from recipebuild_tokenizer import RBTokenizer
from transformers import (CONFIG_MAPPING, MODEL_FOR_MASKED_LM_MAPPING,
                          AutoConfig, AutoModelForMaskedLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, HfArgumentParser,
                          Trainer, TrainingArguments, is_torch_tpu_available,
                          set_seed)
from transformers.trainer_utils import get_last_checkpoint
# from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils import send_example_telemetry
from transformers.utils.versions import require_version

## custom load

VOCAB_CONFIG = "ingr_only"  # 'ingr_only' or 'ingr_title' (ing_title -> memory error)
# VOCAB_CONFIG    =   'ingr_title' # 'ingr_only' or 'ingr_title'
CONFIG_PATH = "/media/ssd/dh/projects/ing_mlm/config.json"

ingt_config = bootstrap.IngTConfig(vocab=VOCAB_CONFIG, path=CONFIG_PATH)


logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

class args:
    data_folder = '/media/ssd/dh/projects/ing_mlm/test_output'
    # model_name_or_path = '/media/ssd/dh/projects/ing_mlm/checkpoints/v1-ing-only_2023-03-08-07-24/checkpoint-150000'
    model_name_or_path = '/media/ssd/dh/projects/ing_mlm/checkpoints/v1-ing-only_2023-03-06-20-51/checkpoint-26000'
args

args.data_folder

ingt_tokenizer = RBTokenizer(ingt_config)
ingt_tokenizer.load()
tokenizer = ingt_tokenizer.tokenizer

config = AutoConfig.from_pretrained(args.model_name_or_path)
config

model = AutoModelForMaskedLM.from_pretrained(
            args.model_name_or_path,
            config=config,
        )

import pandas as pd

tmp_path_2 = "/media/ssd/dh/projects/ing_mlm/data_folder/processed"

f = open(f"{tmp_path_2}/v1_ing_only/test.txt", 'r')

#test = pd.read_table(f"{tmp_path_2}/v1_ing_only/test.txt", names = ['ingredeint'])

##### 첫 번째 MASK

line_list = []
q_list=[]
answer_list = []

pred_list = []
from tqdm import tqdm

with open(f"{tmp_path_2}/v1_ing_only/test.txt", 'r') as f:
    for idx, line in tqdm(enumerate(f)) :
        # print(idx)
        #if idx >= 7000:
        #    break
        #print(line)
        line = line.strip()
        if len(line) == 0 :
            continue

        line_list.append(line)
        word_list = line.split()    
        answer_idx = 0
        answer = word_list[answer_idx]
        q = line.replace(answer, "[MASK]")
        q_list.append(q)
        answer_list.append(answer)

df = pd.DataFrame()
df['original'] = line_list
df['q'] = q_list
df['answer'] = answer_list
df['ori_sp'] = df['original'].apply(lambda x : x.split(' '))
df['ori_len'] = df['ori_sp'].apply(lambda x : len(x))
df['ori_tokenized'] = df['original'].apply(lambda x : tokenizer(str(x),  return_tensors="pt"))
print(len(df))
df

pred_list = []

_topk_num = 10
_pred_topk_list = [[] for i in range(_topk_num)]

from tqdm import tqdm
for _tokenized in tqdm(df['ori_tokenized']):
    #  _tokenized = {'input_ids': ..., 'token_type_ids': ... , 'attention_mask' , ... }
    _pred = model(**_tokenized)
    pred_list.append(_pred)
    
    _res = tokenizer.convert_ids_to_tokens(torch.topk(_pred.logits[0][0], _topk_num).indices)
    for _i, _token in enumerate(_res):
        _pred_topk_list[_i].append(_token)
    # print(_res)
    # print(row)
    

for _i, _pred_tokens in enumerate(_pred_topk_list):
    df[f'top_{_i}'] = _pred_tokens
    
import pickle
import pandas as pd

# 데이터 저장
#df.to_pickle('/media/ssd/dh/projects/ing_mlm/tester/non_backup/df_first_ckpt26.pkl')

# 데이터 저장
df.to_csv('/media/ssd/dh/projects/ing_mlm/tester/non_backup/df_first_ckpt26.txt')