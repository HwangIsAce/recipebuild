import argparse
import bootstrap
from recipebuild_tokenizer import RBTokenizer
import torch
from tqdm import tqdm
import pandas as pd
from collections import Counter
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
argparser.add_argument('--masking_location', type=int, default=0, help ='[MASK] location')

args = argparser.parse_args()

if __name__ == '__main__':

    # VOCAB_CONFIG = "ingr_title"
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
    inference_data_ingr_count = []
    exception_list = []

    with open("/home/donghee/projects/mlm/tester/inference_data_ingr_count.txt") as f:
        lines = f.read().splitlines()

    for line in lines:
        inference_data_ingr_count.append(int(line))

    with open(f"{args.data_path}", 'r') as f:
        for idx, line in enumerate(zip(f, inference_data_ingr_count)):
            
            line_ = line[0].strip()

            if len(line_) == 0:
                continue

            line_list.append(line_)
            word_list = line_.split()

            # 예외처리1
            answer_idx = args.masking_location
            if len(word_list) > args.masking_location:
                answer = word_list[answer_idx] 
            else :
                answer = 'Threki' 

            q = line_.replace(answer, "[MASK]")
            q_list.append(q)
            answer_list.append(answer)

            # 예외처리2
            if answer == 'Threki':
                exception_list.append(0)
                continue
            elif args.masking_location < line[1]:
                exception_list.append(0) 
            else :
                exception_list.append(1) 

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
        _logit = _pred.logits[0][answer_idx].detach().cpu()
        del _pred.logits
        del _pred
        del _t
        del _tokenized
        torch.cuda.empty_cache()
        pred_list.append(_logit)
    
        _res = tokenizer.convert_ids_to_tokens(torch.topk(_logit, _topk_num).indices)
        for _i, _token in enumerate(_res):
            _pred_topk_list[_i].append(_token)


        # iids = [aa['input_ids'] for aa in df['q_tokenized'].head(10)   ]
        # for iid in iids:
        #     print(tokenizer.convert_ids_to_tokens(iid[0]))
        # import IPython; IPython.embed(colors="Linux"); exit(1)
        
    for _i, _pred_tokens in enumerate(_pred_topk_list):
        df[f'predicted #{_i}'] = _pred_tokens
        if _i > 8:
            break

    df['errata_1'] = " "
    df['errata_5'] = " "
    df['errata_10'] = " "
    df['errata_20'] = " "

    df = df.copy()

    for k in [1,5,10,20] :
        for idx, value in enumerate(df.index) :
            topk = list(df.iloc[idx, 6:k+6])

            if df['answer'][idx] in topk:
                df['errata_' + str(k)][idx] =1
            else :
                df['errata_' + str(k)][idx] =0

    df['exception'] = exception_list

    df =df[df['exception'] != 1]

    length = len(df)

    HIT_1 = Counter(df['errata_1'])[1] / length
    HIT_5 = Counter(df['errata_5'])[1] / length
    HIT_10 = Counter(df['errata_10'])[1] / length
    HIT_20 = Counter(df['errata_20'])[1] / length

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
    # save_path = '/home/donghee/projects/mlm/tester/test_output'
    save_path = '/home/donghee/projects/mlm/tester/test_output/ingr_title_tag_tokenizer_nMASK'
    # dh : header None 삭제 
    df.to_csv(f'{save_path}/ing_title_tag_125K_with_tagtokenizer_2023_0925-{answer_idx}-{mrr_score}-{HIT_1}-{HIT_5}-{HIT_10}-{length}.tsv', index=False, sep="\t") 

