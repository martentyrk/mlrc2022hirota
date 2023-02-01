import captum

import spacy

import torch
import torchtext
from torchtext.legacy import data
import torch.nn as nn
import torch.nn.functional as F

from torchtext.vocab import Vocab

from captum.attr import LayerIntegratedGradients, TokenReferenceBase, visualization
import csv
import re
import pickle
import random
from nltk import word_tokenize
import nltk
nltk.download('punkt')
import time
import argparse
import numpy as np
import os
import pprint
from nltk.tokenize import word_tokenize
from io import open
import sys
import json
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm, trange
from operator import itemgetter
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import pandas as pd
from lstm_leakage import main as leak_main


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default='captioning', type=str)
    parser.add_argument("--cap_model", default='sat', type=str)
    parser.add_argument("--gender_or_race", default='gender', type=str)
    parser.add_argument("--calc_ann_leak", default=False, type=bool)
    parser.add_argument("--calc_model_leak", default=False, type=bool)
    parser.add_argument("--test_ratio", default=0.1, type=float)
    parser.add_argument("--balanced_data", default=True, type=bool)
    parser.add_argument("--mask_gender_words", default=True, type=bool)
    parser.add_argument("--save_preds", default=False, type=bool)
    parser.add_argument("--use_glove", default=False, type=bool)
    parser.add_argument("--save_model_vocab", default=False, type=bool)
    parser.add_argument("--align_vocab", default=True, type=bool)
    parser.add_argument("--mask_bias_source", default='', type=str, help='obj or person or both or none')
    parser.add_argument("--return_model", default=False, type=bool)
    parser.add_argument("--num_of_sentences", default=0, type=int)

    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_epochs", default=20, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--save_model", default=False, type=bool)
    parser.add_argument("--workers", default=1, type=int)

    parser.add_argument("--embedding_dim", default=100, type=int)
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--output_dim", default=1, type=int)
    parser.add_argument("--n_layers", default=2, type=int)
    parser.add_argument("--bidirectional", default=True, type=bool)
    parser.add_argument("--dropout", default=0.5, type=float)
    parser.add_argument("--pad_idx", default=0, type=int)
    parser.add_argument("--fix_length", default=False, type=bool)

    return parser
    

def interpret_sentence(model, sentence, text_lengths, vis_data_records, text_original, model_labels, lig, device, token_reference, min_len = 15, label = 0, att_dict = {}):
    text = [text_original.vocab.itos[t] for t in sentence.permute(1,0)]

    indexed = [text_original.vocab.stoi[t] for t in text]
    model.zero_grad()
    input_indices = torch.tensor(indexed, device=device)
    input_indices = input_indices.unsqueeze(0)

    input_indices = sentence

    # logit 
    pred = model(input_indices, text_lengths)[0]
    

    # logit to prob
    probs = torch.sigmoid(pred).item()
    pred_ind = round(probs)
    #print("XXX")
    # generate reference indices for each sample
    reference_indices = token_reference.generate_reference(text_lengths, device=device).unsqueeze(0)
    # compute attributions and approximation delta using layer integrated gradients
    attributions_ig, delta = lig.attribute(input_indices, reference_indices, \
                                           additional_forward_args=text_lengths,
                                           n_steps=500, return_convergence_delta=True)

    add_attributions_to_visualizer(attributions_ig, text, probs, pred_ind, label, delta, vis_data_records, model_labels, att_dict)

    return vis_data_records

def add_attributions_to_visualizer(attributions, text, pred, pred_ind, label, delta, vis_data_records, model_label, att_dict):
    attributions = attributions.sum(dim=2).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    attributions = attributions.cpu().detach().numpy()
    
  
    model_pred = int(model_label.vocab.itos[pred_ind])
    target = int(model_label.vocab.itos[label])
    
    if model_pred == target:
      if target == 0:
        att_dict['male_att_score'].append(attributions.sum())
      elif target == 1:
        att_dict['female_att_score'].append(attributions.sum())


    # storing couple samples in an array for visualization purposes
    vis_data_records.append(visualization.VisualizationDataRecord(
                            attributions,
                            pred,
                            model_label.vocab.itos[pred_ind],
                            model_label.vocab.itos[label],
                            model_label.vocab.itos[0],
                            attributions.sum(),
                            text,
                            delta))
    
    
def run_sentence_interpretation(args, model_dict, model_att, model, text, model_labels, lig, device, token_reference):
  # accumalate couple samples in this array for visualization purposes
    model_data_records = []
    
    counter = 0


    keys = model_dict.keys()
    num_of_sentences = args.num_of_sentences
    
    if args.num_of_sentences == 0:
        num_of_sentences = len(keys)
        print('Iterating over:', num_of_sentences, 'samples.')

    for key in keys:      
        counter += 1
        model_text, model_text_length = model_dict[key][0]

        model_text = model_text.permute(1,0)
        label = model_dict[key][1]
        
        sent_list = []

        interpret_sentence(
            model, model_text, model_text_length, text_original=text ,model_labels=model_labels,
            lig=lig, vis_data_records=model_data_records, att_dict=model_att, label=label, device=device, token_reference=token_reference
            )

        if counter == num_of_sentences:
            break

    with open('attributions/LSTM/'+str(args.cap_model)+'_seed'+ str(args.seed) +'attributions.pickle', 'wb') as handle:
        pickle.dump(model_att, handle, protocol=pickle.HIGHEST_PROTOCOL)


def forward_with_sigmoid(input, text_lengths=None):
    return torch.sigmoid(model(input, text_lengths=text_lengths)[0])


def main(args, model_dict, model_att, model, TEXT, LABEL, lig, device, token_reference):
    run_sentence_interpretation(args, model_dict, model_att, model, TEXT, LABEL, lig, device, token_reference)
    
    
    
if __name__ == "__main__":
    parser = get_parser()
    args, unknown = parser.parse_known_args()
    print("---Start---")
    print('Seed:', args.seed)
    print("Task:", args.task)
    if args.task == 'captioning' and args.calc_model_leak:
        print("Captioning model:", args.cap_model)
    print("Gender or Race:", args.gender_or_race)

    if args.calc_ann_leak:
        print('Align vocab:', args.align_vocab)
        if args.align_vocab:
            print('Vocab of ', args.cap_model)
    print()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_dict, TEXT, LABEL, IMID = leak_main(args)
    
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

    # PADDING NEUTRAL
    token_reference = TokenReferenceBase(reference_token_idx=PAD_IDX)
    
    lig = LayerIntegratedGradients(forward_with_sigmoid, model.embedding)
    model_att = {
    'female_att_score':[],
    'male_att_score':[],
    }

    main(args, model_dict, model_att, model, TEXT, LABEL, lig, device, token_reference)

    




        