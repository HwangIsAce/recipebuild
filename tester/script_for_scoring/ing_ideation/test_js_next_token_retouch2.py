import argparse
import bootstrap
from recipebuild_tokenizer import RBTokenizer
import torch
from tqdm import tqdm
import pandas as pd
import transformers
from fractions import Fraction
from datasets import load_dataset
from transformers import (CONFIG_MAPPING, MODEL_FOR_MASKED_LM_MAPPING,
                          AutoConfig, AutoModelForMaskedLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, HfArgumentParser,
                          Trainer, TrainingArguments,is_torch_tpu_available,
                          set_seed)


kk = open('/home/donghee/projects/mlm/tester/exclude_list2.txt')
lines = kk.readlines()

exclude_list = []
for line in lines:
    exclude_list.append(line.replace("\n",""))


argparser = argparse.ArgumentParser()
argparser.add_argument('--data_path', type=str, default='', help='dataset path')
argparser.add_argument('--model_name_or_ckpt_path', type=str, default='', help='model name or ckpt path')
argparser.add_argument('--topk', type=int, default=1, help='model predict length(count)')
args = argparser.parse_args()


_MASK_NUM = 4

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

    IDX = [] # 
    SUB_IDX = [] # 
    SEN_ORIGINAL = [] # 
    MODEL_INPUT = [] #
    ANSWER = [] # 
    PRED = [] 
    SCORE = [] #
    
    cnt1 = 0
    with open(f"{args.data_path}", 'r') as f:
        for idx, line in tqdm(enumerate(f)):
            line = line.strip()

            if len(line) == 0:
                continue

            split_line = line.split('[SEP]')
            list_ing = split_line[0]
            list_tag = split_line[1]
            list_title = split_line[2]

            answer = list_ing.split(' ')
            # ANSWER.append(answer)

            # SEN_ORIGINAL.append(line)
            
            cnt1 += 1
            # IDX.append(cnt1)

            temp = []; answer = []; list_model_input = []; cnt2 = 0
            score = []; 
            for i, _ing in enumerate(list_ing.split(' ')):
                IDX.append(cnt1)
                SEN_ORIGINAL.append(line)
                ANSWER.append(_ing)

                answer_cnt = 0

                
                temp.append(_ing)
                masking_sen_answer = list_tag + ' [SEP] ' + list_title + ' [SEP] ' + ' '.join(temp)

                temp[i] = '[MASK]'

                model_input = list_tag + ' [SEP] ' + list_title + ' [SEP] ' + ' '.join(temp)
                MODEL_INPUT.append(model_input)
                cnt2 += 1
                SUB_IDX.append(cnt2)

                model_output = []

                _device = torch.device('cuda:1')
                model.to(_device)

                _topk_num = args.topk
                pred = []

                _tokenized = tokenizer(str(model_input), return_tensors="pt")
                _t = {k: v.to(_device) for k, v in _tokenized.items()}

                _pred = model(**_t)

                _MASK_IDX = torch.where(_t['input_ids'][0]==_MASK_NUM)[0][0].detach().cpu()
                _logit = _pred.logits[0][_MASK_IDX].detach().cpu()

                del _pred.logits
                del _pred
                del _t
                del _tokenized
                torch.cuda.empty_cache()
                model_output.append(_logit)
                
                _res = tokenizer.convert_ids_to_tokens(torch.topk(_logit, _topk_num).indices)
                                
                start =3
                while  len(list(set(_res)-set(exclude_list))) < 3 :
                    _res = list(set(_res)-set(exclude_list))

                    end = start + (3-len(_res))

                    tmp = tokenizer.convert_ids_to_tokens(torch.sort(_logit)[1][::1].flip(dims=(0,))[start:end])
                    _res.extend(tmp)

                    start = start + end

                for _i, _token in enumerate(_res):
                    pred.append(_token)
                PRED.append(pred)

                if _ing in pred:
                    answer_cnt += 1

                SCORE.append(answer_cnt)

                temp[i] = _ing
            
    df = pd.DataFrame()

    df['IDX'] = IDX
    df['SUB_IDX'] = SUB_IDX
    df['ORIGINAL'] = SEN_ORIGINAL
    df['MODEL_INPUT'] = MODEL_INPUT
    df['ANSWER'] = ANSWER
    df['PRED'] = PRED
    df['SCORE'] = SCORE

    save_path =  '/home/donghee/projects/mlm/tester/test_output'
    df.to_csv(f'{save_path}/{args.topk}-next_token_predict_2023_1026_2.tsv', index=False, sep="\t")

            
