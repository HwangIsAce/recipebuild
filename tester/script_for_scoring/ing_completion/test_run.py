# 필요한 라이브러리 import 
import argparse
import logging
import math
import os
import random
import sys
# 추가적인 import 
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from itertools import chain
from typing import Optional

## custom load
import bootstrap
import datasets
import evaluate
import numpy as np
import pandas as pd
# custom config load
import torch
import torch.backends.cudnn as cudnn  # 랜덤 시드 고정
import transformers
from datasets import load_dataset
from recipebuild_tokenizer import RBTokenizer
from tqdm import tqdm
from transformers import (CONFIG_MAPPING, MODEL_FOR_MASKED_LM_MAPPING,
                          AutoConfig, AutoModelForMaskedLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, HfArgumentParser,
                          Trainer, TrainingArguments, is_torch_tpu_available,
                          set_seed)
from transformers.trainer_utils import get_last_checkpoint
# from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils import send_example_telemetry
from transformers.utils.versions import require_version

import wandb

#파이토치의 랜덤시드 고정
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0) # gpu 1개 이상일 때 

# 넘파이 랜덤시드 고정
np.random.seed(0)

#CuDNN 랜덤시드 고정
cudnn.benchmark = False
cudnn.deterministic = True # 연산 처리 속도가 줄어들어서 연구 후반기에 사용하자

# 파이썬 랜덤시드 고정
random.seed(0)

#data_path->  '/media/ssd/dh/projects/ing_mlm/test_output'
#model_name_or_checkpoint_path->  '/media/ssd/dh/projects/ing_mlm/checkpoints/v1-ing-only_2023-03-06-20-51/checkpoint-26000' # 사용할 모델

#python test.py --project_name test --data_path "/media/ssd/dh/projects/ing_mlm/test_output" --model_name_or_checkpoint_path "/media/ssd/dh/projects/ing_mlm/checkpoints/v1-ing-only_2023-03-06-20-51/checkpoint-10000" --mask_position first last random top_0_33p top_33p_66p top_66p_100p
#python test.py --project_name test --data_path "/media/ssd/dh/projects/ing_mlm/test_output" --model_name_or_checkpoint_path "/media/ssd/dh/projects/ing_mlm/checkpoints/v1-ing-only_2023-03-08-07-24/checkpoint-50000" --mask_position first last random top_0_33p top_33p_66p top_66p_100p
def arg_parse():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--project_name', type=str, default='', help='wandb project name')
    argparser.add_argument('--data_path', type=str, default='', help='dataset path')
    argparser.add_argument('--model_name_or_checkpoint_path', type=str,
                           default='', help='model name or checkpoint path')
    argparser.add_argument('--mask_position', type=str, default='', nargs = 6,
                           help='where to mask : first, last, random, top_0_33p, top_33p_66p, top_66p_100p')  # 아 이러면 귀찮아지는구나 -> 수정하기
    return argparser.parse_args()

args = arg_parse()
    
VOCAB_CONFIG = "ingr_only"  # 'ingr_only' or 'ingr_title' (ing_title -> memory error)
CONFIG_PATH = "/media/ssd/dh/projects/ing_mlm/config.json"
ingt_config = bootstrap.IngTConfig(vocab=VOCAB_CONFIG, path=CONFIG_PATH)
    
logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
    
data_folder = args.data_path 
model_name_or_path = args.model_name_or_checkpoint_path
    
ingt_tokenizer = RBTokenizer(ingt_config)
ingt_tokenizer.load()
tokenizer = ingt_tokenizer.tokenizer
    
config = AutoConfig.from_pretrained(model_name_or_path)
model = AutoModelForMaskedLM.from_pretrained(
        model_name_or_path,
        config=config,
    )

def split_list(lst): # 리스트 3등분해서 반환
    div3 = len(lst)//3    
    if len(lst) % 3 == 2 : 
        return lst[:div3+1], lst[div3+1:-div3], lst[-div3:]
    return lst[:div3], lst[div3:-div3], lst[-div3:]    

def inference_and_mk_tsv(ver) : #[MASK] 에 대한 모델의 pred 와 ans, q 등을 칼럼으로 가지는 dataframe 반환
    line_list = []
    q_list=[]
    answer_list = []
    pred_list = []
    
    with open(f"/media/ssd/dh/projects/ing_mlm/data_folder/processed/v1_ing_only/test.txt", 'r') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if len(line) == 0 :
                continue
            

            line_list.append(line)
            word_list = line.split()  
            if ver == 'first' : # 맨 앞 [MASK]
                answer_idx = 0
                answer = word_list[answer_idx]
            elif ver == 'last' : # 맨 뒤 [MASK]
                answer_idx = -1
                answer = word_list[answer_idx]
            elif ver == 'random' : # RANDOM [MASK]
                answer_idx = random.choice(range(len(word_list)))
                answer = word_list[answer_idx]
                
            elif ver == 'top_0_33p' : # 구간 상위 33% 에서 랜덤 [MASK]
                word_list_chuncked = split_list(word_list)
                answer_idx = 0 
                answer = random.choice(word_list_chuncked[answer_idx])
            elif ver == 'top_33p_66p' : # 구간 중위 33% 에서 랜덤 [MASK]
                word_list_chuncked = split_list(word_list)
                answer_idx = 1
                answer = random.choice(word_list_chuncked[answer_idx])
            elif ver == 'top_66p_100p' : # 구간 하위 33% 에서 랜덤 [MASK]
                word_list_chuncked = split_list(word_list)
                answer_idx = 2 
                answer = random.choice(word_list_chuncked[answer_idx])
            q = line.replace(answer, "[MASK]")
            q_list.append(q)
            answer_list.append(answer)
            
    df = pd.DataFrame()
    df['original'] = line_list
    df['q'] = q_list
    df['answer'] = answer_list
    df['ori_sp'] = df['original'].apply(lambda x : x.split(' '))
    df['ori_len'] = df['ori_sp'].apply(lambda x : len(x))
    df['q_tokenized'] = df['q'].apply(lambda x : tokenizer(str(x),  return_tensors="pt"))

    pred_list = []

    _topk_num = 10
    _pred_topk_list = [[] for i in range(_topk_num)]
    _device = torch.device('cuda:1')

    model.to(_device)

    for _tokenized in tqdm(df['q_tokenized']):
        #  _tokenized = {'input_ids': ..., 'token_type_ids': ... , 'attention_mask' , ... }
        _t = {k:v.to(_device)for k,v in _tokenized.items()}
    
        _pred = model(**_t)
        _logit = _pred.logits[0][0].detach().cpu()
        del _pred.logits
        del _pred
        del _t
        del _tokenized
        torch.cuda.empty_cache()
        pred_list.append(_logit)
    
        _res = tokenizer.convert_ids_to_tokens(torch.topk(_logit, _topk_num).indices)
        for _i, _token in enumerate(_res):
            _pred_topk_list[_i].append(_token)
        
    for _i, _pred_tokens in enumerate(_pred_topk_list):
        df[f'top_{_i}'] = _pred_tokens
        
    return df

def inference_score(dataframe_) : # 이러면 무조건 데이터프레임 형태가 변하면 안되는데 더 좋은 방법 생각해보자. -> 수정하기 -> wandb 가 해결방법이구나 
    dataframe_['errata_1'] = " "
    dataframe_['errata_3'] = " "
    dataframe_['errata_5'] = " "
    dataframe_['errata_10'] = " "
    
    for k in [1,3,5,10] : 
        for idx, value in enumerate(dataframe_.index) : 
            #print(dataframe_['answer'][idx])
            topk = list(dataframe_.iloc[idx, 6:k+6]) 
            #print(topk)
            if dataframe_['answer'][idx] in topk : 
                dataframe_['errata_' + str(k)][idx] = 1 
                
            else :
                dataframe_['errata_' + str(k)][idx] = 0 
                
    
    length = len(dataframe_)
    
    acc_1 = dataframe_['errata_1'].value_counts()[1] / length
    acc_3 = dataframe_['errata_3'].value_counts()[1] / length
    acc_5 = dataframe_['errata_5'].value_counts()[1] / length
    acc_10 = dataframe_['errata_10'].value_counts()[1] / length
    
    return acc_1, acc_3, acc_5, acc_10
   

def main():
        
    for mp in args.mask_position :
        # inference 해서 tsv 만들기
        df = inference_and_mk_tsv(mp)
    
        # acc@1 acc@3 acc@5 acc@10 점수 내기
        acc_1, acc_3, acc_5, acc_10 = inference_score(df)
        print('acc_1 : {}, acc_3 : {}, acc_5 : {}, acc_10 : {}'.format(acc_1, acc_3, acc_5, acc_10))
        
        # ckpt 맨 마지막 경로(?) 가져오기 # m_last_path : m 의 경로의 마지막 폴더를 뜻함
        m_last_path = os.path.basename(os.path.normpath(model_name_or_path))
        m_last_path_ckpt_num = m_last_path.split('-')
        
        # wandb 

        exp_table = wandb.Table(dataframe=df)
    
        run = wandb.init(project='test_exp')
        
        wandb.config = {
        "epochs": 100, 
        "learning_rate": 0.001, 
        "batch_size": 128 
        }
        
        run.name = f'{m_last_path_ckpt_num[0]}-{int(m_last_path_ckpt_num[1]) :07d}-{mp}' # 이거 되나?
        
        run.log({'acc@1' : acc_1, 'acc@3' : acc_3, 'acc@5' : acc_5, 'acc@10' : acc_10})
    
    
        # tsv 파일 저장
        save_path = '/media/ssd/dh/projects/ing_mlm/processing/'
        # dh : header None 삭제 
        df.to_csv(f'{save_path}/test_2023_0324/{m_last_path}-{mp}.tsv', index=False, sep="\t") 
    
        #artifact
        with wandb.init(project="test_exp", dir=f"/media/ssd/dh/projects/ing_mlm/processing/test_2023_0324/{m_last_path}", job_type="load-data") as run: 

            # 🏺 create our Artifact
            exp_artifact = wandb.Artifact(
                "exp_artifact-data", type="dataset",
                description="test")
        
            # 🐣 Store a new file in the artifact, and write something into its contents.
            exp_artifact.add_file(f'{save_path}/test_2023_0324/{m_last_path}-{mp}.tsv') 
            
            # ✍️ Save the artifact to W&B.
            run.log_artifact(exp_artifact)
    
        
    ##############################################################################################################################
    # # inference 해서 tsv 만들기
    # df = inference_and_mk_tsv(args.mask_position)
    
    # # acc@1 acc@3 acc@5 acc@10 점수 내기
    # acc_1, acc_3, acc_5, acc_10 = inference_score(df)
    # print('acc_1 : {}, acc_3 : {}, acc_5 : {}, acc_10 : {}'.format(acc_1, acc_3, acc_5, acc_10))
    
    # # wandb 

    # exp_table = wandb.Table(dataframe=df)
    
    # run = wandb.init(project='test_exp')
        
    # wandb.config = {
    # "epochs": 100, 
    # "learning_rate": 0.001, 
    # "batch_size": 128 
    # }
        
    # run.log({'acc@1' : acc_1, 'acc@3' : acc_3, 'acc@5' : acc_5, 'acc@10' : acc_10})
    
    # # tsv 파일 저장
    # save_path = '/media/ssd/dh/projects/ing_mlm/processing/'
    # df.to_csv(f'{save_path}/test_2023_0324/{args.mask_position}.tsv', index=False, header=None, sep="\t") # 이거 되나?
    
    # #artifact
    # with wandb.init(project="test_exp", dir="/media/ssd/dh/projects/ing_mlm/processing/test_2023_0324/wandb", job_type="load-data") as run:

    #     # 🏺 create our Artifact
    #     exp_artifact = wandb.Artifact(
    #         "exp_artifact-data", type="dataset",
    #         description="test")
        
    #     # 🐣 Store a new file in the artifact, and write something into its contents.
    #     exp_artifact.add_file(f'{save_path}/test_2023_0324/{args.mask_position}.tsv')
            
    #     # ✍️ Save the artifact to W&B.
    #     run.log_artifact(exp_artifact)

    
if __name__ == '__main__':
    print('start')
    main()
    
