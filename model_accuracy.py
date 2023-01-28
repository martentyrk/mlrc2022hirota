
import pickle
import random
from nltk import word_tokenize
import nltk
nltk.download('punkt')
import time
import argparse
import numpy as np
from nltk.tokenize import word_tokenize
from io import open
import sys
import json
from tqdm import tqdm, trange
from operator import itemgetter
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
from pycocoevalcap.cider.cider import Cider


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default='captioning', type=str)
    parser.add_argument("--cap_model", default='sat', type=str)
    parser.add_argument("--gender_or_race", default='gender', type=str)
    parser.add_argument("--test_ratio", default=0.1, type=float)
    parser.add_argument("--balanced_data", default=True, type=bool)
    parser.add_argument("--mask_gender_words", default=True, type=bool)
    parser.add_argument("--use_glove", default=False, type=bool)
    parser.add_argument("--save_model_vocab", default=False, type=bool)
    parser.add_argument("--align_vocab", default=True, type=bool)
    parser.add_argument("--mask_bias_source", default='', type=str, help='obj or person or both or none')


    return parser

###
# Calculate the meteor score based on captions
# Works by matching/aligning the annotation words with model
# captions.
# Params to the method
# annotations: a list of lists with all annotations.
# model_captions: a list of all model captions for a specific model.
###

def calculate_meteor(annotations, model_captions):
    all_scores = []
    
    assert(annotations.keys() == model_captions.keys())
    imgIds = annotations.keys()
    for id in imgIds:
        all_scores.append(meteor_score(annotations[id], model_captions[id][0]))
    
    return sum(all_scores) / len(all_scores)

def calculate_bleu(annotations, model_captions):
    all_scores = []
    
    assert(annotations.keys() == model_captions.keys())
    imgIds = annotations.keys()
    
    for id in imgIds:
        all_scores.append(sentence_bleu(annotations[id], model_captions[id][0], weights=(0.25, 0.25, 0.25, 0.25)))
    
    
    return sum(all_scores) / len(all_scores)

def prepare_human_ann(human_annotations):
    all_annotations = {}
        
    for entry in human_annotations:
        all_annotations[entry['img_id']] = entry['caption_list']
    
    return all_annotations

def prepare_model_captions(model_captions):
    all_captions = {}
    
    for entry in model_captions:
        all_captions[entry['img_id']] = [entry['pred']]
    
    
    return all_captions
    


def main(args):
    gender_human_annotations = pickle.load(open('bias_data/Human_Ann/gender_obj_cap_mw_entries.pkl', 'rb')) # Human captions
    
    #Select captioning model
    if args.cap_model == 'nic':
        selected_cap_gender_entries = pickle.load(open('bias_data/Show-Tell/gender_val_st10_th10_cap_mw_entries.pkl', 'rb'))
        
    elif args.cap_model == 'sat':
        selected_cap_gender_entries = pickle.load(open('bias_data/Show-Attend-Tell/gender_val_sat_cap_mw_entries.pkl', 'rb'))
        
    elif args.cap_model == 'fc':
        selected_cap_gender_entries = pickle.load(open('bias_data/Att2in_FC/gender_val_fc_cap_mw_entries.pkl', 'rb'))
        
    elif args.cap_model == 'att2in':
        selected_cap_gender_entries = pickle.load(open('bias_data/Att2in_FC/gender_val_att2in_cap_mw_entries.pkl', 'rb'))
        
    elif args.cap_model == 'updn':
        selected_cap_gender_entries = pickle.load(open('bias_data/UpDn/gender_val_updn_cap_mw_entries.pkl', 'rb'))
        
    elif args.cap_model == 'transformer':
        selected_cap_gender_entries = pickle.load(open('bias_data/Transformer/gender_val_transformer_cap_mw_entries.pkl', 'rb'))
        
    elif args.cap_model == 'oscar':
        selected_cap_gender_entries = pickle.load(open('bias_data/Oscar/gender_val_cider_oscar_cap_mw_entries.pkl', 'rb'))
        
    elif args.cap_model == 'nic_equalizer':
        selected_cap_gender_entries = pickle.load(open('bias_data/Woman-Snowboard/gender_val_snowboard_cap_mw_entries.pkl', 'rb'))
        
    elif args.cap_model == 'nic_plus':
        selected_cap_gender_entries = pickle.load(open('bias_data/Woman-Snowboard/gender_val_baselineft_cap_mw_entries.pkl', 'rb'))

    #df_humancaptions = pd.DataFrame(gender_human_annotations, columns=['img_id', 'caption_list'])
    #df_modelcaptions = pd.DataFrame(selected_cap_gender_entries, columns=['img_id', 'pred'])
    
    #df_humancaptions = df_humancaptions.sort_values('img_id')
    #df_modelcaptions = df_modelcaptions.sort_values('img_id')
    

    #sorted_human_captions = list(df_humancaptions['caption_list'])
    #sorted_model_captions = list(df_modelcaptions['pred'])
    
    all_scores = {
        'meteor': 0,
        'rougel': 0,
        'bleu4': 0,
        'cider': 0
    }
    dict_model_capt = prepare_model_captions(selected_cap_gender_entries)
    dict_human_ann = prepare_human_ann(gender_human_annotations)
        
    #METEOR
    all_scores['meteor'] = calculate_meteor(dict_human_ann, dict_model_capt)
    
    #ROUGE-L
    all_scores['rougel'] = Rouge().compute_score(dict_human_ann, dict_model_capt)
    
    #BLEU-4
    all_scores['bleu4'] = calculate_bleu(dict_human_ann, dict_model_capt)
    
    #CIDER
    all_scores['cider'], _ = Cider().compute_score(dict_human_ann, dict_model_capt)
    
    
    print(all_scores)

        

if __name__ == "__main__":
    parser = get_parser()
    args, unknown = parser.parse_known_args()
    print("---Start---")
    
    print("Task:", args.task)
    
    print("Gender or Race:", args.gender_or_race)
    print()

    main(args)