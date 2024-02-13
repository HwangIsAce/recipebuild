import argparse
import bootstrap
from recipebuild_tokenizer import RBTokenizer
import torch
from tqdm import tqdm
import pandas as pd
import transformers
from datasets import load_dataset
from transformers import (CONFIG_MAPPING, MODEL_FOR_MASKED_LM_MAPPING,
                          AutoConfig, AutoModelForMaskedLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, HfArgumentParser,
                          Trainer, TrainingArguments, is_torch_tpu_available,
                          set_seed)

argparser = argparse.ArgumentParser()
argparser.add_argument('--data_path', type=str, default='', help='dataset path')
argparser.add_argument('--model_name_or_ckpt_path', type=str, default='', help='model name or ckpt path')

args = argparser.parse_args()

if __name__ == '__main__':

    VOCAB_CONFIG = "ingr_title_tag"
    CONFIG_PATH = "/home/donghee/projects/mlm/config.json"
    ingt_config = bootstrap.IngTConfig(vocab=VOCAB_CONFIG, path=CONFIG_PATH)

    ingt_tokenizer = RBTokenizer(ingt_config)
    ingt_tokenizer.load()
    tokenizer = ingt_tokenizer.tokenizer

    config = AutoConfig.from_pretrained(args.model_name_or_ckpt_path)

    model = AutoModelForMaskedLM.from_pretrained(
        args.model_name_or_ckpt_path,
        config=config,
    )

    line_list = []
    q_list = []
    answer_list = []
    pred_list = []

    with open(f"{args.data_path}", 'r') as f:
        for idx, line in enumerate(f):
            
            line = line.strip()

            if len(line) == 0:
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
    df['q_tokenized'] = df['q'].apply(lambda x : tokenizer(str(x),  return_tensors="pt"))

    pred_list = []

    _topk_num = 20
    _pred_topk_list = [[] for i in range(_topk_num)]
    _device = torch.device('cuda:1')
    #_device = torch.device('cpu')

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
        df[f'predicted #{_i}'] = _pred_tokens
        if _i > 8:
            break

    df['errata_1'] = " "
    df['errata_5'] = " "
    df['errata_10'] = " "
    df['errata_20'] = " "

    for k in [1,5,10,20] :
        for idx, value in enumerate(df.index) :
            topk = list(df.iloc[idx, 6:k+6])

            if df['answer'][idx] in topk:
                df['errata_' + str(k)][idx] =1
            else :
                df['errata_' + str(k)][idx] =0

    length = len(df)

    HIT_1 = df['errata_1'].value_counts()[1] / length
    HIT_5 = df['errata_5'].value_counts()[1] / length
    HIT_10 = df['errata_10'].value_counts()[1] / length
    HIT_20 = df['errata_20'].value_counts()[1] / length

    mrr = []
    for idx in df.index :
        cnt =0
        tmp = df['answer'].loc[idx]
        for col in df.loc[idx]['predicted #0':'predicted #9'] :
            cnt +=1
            if tmp == col:
                mrr.append(1/cnt)
                break

    mrr_score = sum(mrr) / len(df)

    # tsv 파일 저장
    save_path = '/home/donghee/projects/mlm/tester/test_output'
    # dh : header None 삭제 
    df.to_csv(f'{save_path}/ing_title_tag_test_125K_with_tagtokenizer_2023_0910-{mrr_score}-{HIT_1}-{HIT_5}-{HIT_10}-{HIT_20}.tsv', index=False, sep="\t") 

