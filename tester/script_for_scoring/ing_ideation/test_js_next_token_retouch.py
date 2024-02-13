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

    my_answer_sentence = []
    my_answer_ing_list = []
    my_model_input = []
    my_pred_topk_list = []
    my_answer_ing_count = []
    my_index_per_row = [] 
    my_score = []
    with open(f"{args.data_path}", 'r') as f:
        score_list = []

        for idx, line in enumerate(f):
            ##
            model_output_list = []

            line = line.strip()
            
            if len(line) == 0:
                continue

            line_split = line.split('[SEP]')
            ing_list = line_split[0]
            tag_list = line_split[1]
            title_list = line_split[2]

            answer_ing_list = ing_list.split(' ')

            tag_list_count = len(tag_list.split(' '))
            title_list_count = len(title_list.split(' '))
            index_for_logits = tag_list_count + title_list_count

            tag_list.split(' ')

            answer_sentence = tag_list + ' [SEP] ' + title_list + ' [SEP] ' + ing_list

            answer_ing_count = len(ing_list.split(' '))

            _ing_accumulate_list = []

            for i, _ing in enumerate(ing_list.split(' ')):
                
                cnt = 0

                _ing_accumulate_list.append(_ing)
                answer_sen = tag_list + ' [SEP] ' + title_list + ' [SEP] ' + ' '.join(_ing_accumulate_list)

                _ing_accumulate_list[i] = '[MASK]'
                answer_ing = _ing
                model_input = tag_list + ' [SEP] ' + title_list + ' [SEP] ' + ' '.join(_ing_accumulate_list)

                model_output = []

                _device = torch.device('cuda:1')
                model.to(_device)

                _topk_num = args.topk
                # _pred_topk_list = [[] for z in range(_topk_num)]
                _pred_topk_list = []

                # model predict
                _tokenized = tokenizer(str(model_input), return_tensors="pt")

                _t = {k:v.to(_device) for k, v in _tokenized.items()}

                _pred = model(**_t)

                _MASK_IDX = torch.where(_t['input_ids'][0] == _MASK_NUM)[0][0].detach().cpu()
                # _logit = _pred.logits[0][index_for_logits+2+i].detach().cpu()
                _logit = _pred.logits[0][_MASK_IDX].detach().cpu()
                # import IPython; IPython.embed(color="Linux"); exit(1)
                del _pred.logits
                del _pred
                del _t
                del _tokenized
                torch.cuda.empty_cache()
                model_output.append(_logit)

                _res = tokenizer.convert_ids_to_tokens(torch.topk(_logit, _topk_num).indices)

                for _i, _token in enumerate(_res):
                    # _pred_topk_list[_i].append(_token)
                    _pred_topk_list.append(_token)

                if answer_ing in _pred_topk_list:
                    cnt += 1

                _ing_accumulate_list[i] = _ing
                                
                model_output_list.append(_pred_topk_list)
                my_model_input.append(model_input)
                my_index_per_row.append(f"{i+1}개")

                score = cnt/answer_ing_count

                score_list.append(score)

                # if answer_ing == 'water':
                #     import IPython; IPython.embed(color="Linux"); exit(1)

            my_answer_sentence.append(answer_sentence)
            my_pred_topk_list.append(model_output_list)
            my_answer_ing_count.append(f"총 {answer_ing_count}개")
            my_score = score_list
            my_answer_ing_list.append(answer_ing_list)


            # cnt_list.append(score)
            # print(Fraction(cnt/answer_ing_count))


        df = pd.DataFrame()
        df['answer_sentence'] = my_answer_sentence
        df['answer_ing'] = my_answer_ing_list 
        df['model_output_list'] = my_pred_topk_list
        df['ing_count'] = my_answer_ing_count
        # df['score'] = my_score

        df = df.explode(['answer_ing','model_output_list'], ignore_index=True)
    df['score'] = my_score
    df['model_input'] = my_model_input
    df['input_ing_count'] = my_index_per_row

    # df = df[['answer_sentence', 'model_input', 'answer_ing', 'model_output_list', 'input_ing_count', 'ing_count', 'score']]

    df = df[['answer_sentence', 'model_input', 'answer_ing', 'model_output_list', 'input_ing_count', 'ing_count', 'score']]
    
    save_path = '/home/donghee/projects/mlm/tester/test_output'
    df.to_csv(f'{save_path}/{args.topk}-next_token_predict_2023_1005.tsv', index=False, sep="\t") 


