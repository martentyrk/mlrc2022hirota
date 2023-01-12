import torch
import csv
import spacy
import re
import pickle
import random
import csv
import nltk
nltk.download('punkt')
import argparse
import numpy as np
from io import open
import pickle
from transformers import BertTokenizer

from model import BERT_GenderClassifier
from race_dataset import BERT_ANN_leak_data, BERT_MODEL_leak_data

from bert_utils import make_train_test_split_race, calc_random_acc_score, calc_leak

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default='captioning', type=str)
    parser.add_argument("--cap_model", default='sat', type=str)
    parser.add_argument("--gender_or_race", default='race', type=str)
    parser.add_argument("--calc_ann_leak", default=False, type=bool)
    parser.add_argument("--calc_model_leak", default=False, type=bool)
    parser.add_argument("--calc_race_acc", default=True, type=bool)
    parser.add_argument("--test_ratio", default=0.1, type=float)
    parser.add_argument("--balanced_data", default=True, type=bool)
    parser.add_argument("--mask_race_words", default=False, type=bool)
    parser.add_argument("--freeze_bert", default=False, type=bool)
    parser.add_argument("--store_topk_race_pred", default=False, type=bool)
    parser.add_argument("--topk_race_pred", default=50, type=int)
    parser.add_argument("--calc_score", default=True, type=bool)
    parser.add_argument("--align_vocab", default=True, type=bool)

    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_epochs", default=5, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--optimizer", default='adamw', type=str, help="adamw or adam")
    parser.add_argument("--adam_correct_bias", default=True, type=bool)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.98, type=float, help="0.999:huggingface, 0.98:RoBERTa paper")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer. 1e-8:first, 1e-6:RoBERTa paper")
    parser.add_argument("--weight_decay", default=0.1, type=float, help="Weight deay if we apply some. 0.001:first, 0.01:RoBERTa")
    parser.add_argument("--coco_lk_model_dir", default='/Bias/leakage/', type=str)
    parser.add_argument("--save_model", default=False, type=bool)
    parser.add_argument("--save_model_path", default='/bias-vl/bert.pt', type=str)
    parser.add_argument("--workers", default=1, type=int)

    parser.add_argument("--max_seq_length", default=64, type=int)
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--output_dim", default=1, type=int)
    parser.add_argument("--dropout", default=0.5, type=float)

    return parser



def main(args):
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("device: {} n_gpu: {}".format(device, n_gpu))
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    race_val_obj_cap_entries = pickle.load(open('bias_data/Human_Ann/race_val_obj_cap_entries.pkl', 'rb')) # Human captions

    #Select captioning model
    if args.cap_model == 'nic':
        selected_cap_race_entries = pickle.load(open('bias_data/Show-Tell/race_val_st10_cap_entries.pkl', 'rb'))
    elif args.cap_model == 'sat':
        selected_cap_race_entries = pickle.load(open('bias_data/Show-Attend-Tell/race_val_sat_cap_entries.pkl', 'rb'))
    elif args.cap_model == 'fc':
        selected_cap_race_entries = pickle.load(open('bias_data/Att2in_FC/race_val_fc_cap_entries.pkl', 'rb'))
    elif args.cap_model == 'att2in':
        selected_cap_race_entries = pickle.load(open('bias_data/Att2in_FC/race_val_att2in_cap_entries.pkl', 'rb'))
    elif args.cap_model == 'updn':
        selected_cap_race_entries = pickle.load(open('bias_data/UpDn/race_val_updn_cap_entries.pkl', 'rb'))
    elif args.cap_model == 'transformer':
        selected_cap_race_entries = pickle.load(open('bias_data/Transformer/race_val_transformer_cap_entries.pkl', 'rb'))
    elif args.cap_model == 'oscar':
        selected_cap_race_entries = pickle.load(open('bias_data/Oscar/race_val_cider_oscar_cap_entries.pkl', 'rb'))
    elif args.cap_model == 'nic_equalizer':
        selected_cap_race_entries = pickle.load(open('bias_data/Woman-Snowboard/race_val_snowboard_cap_entries.pkl', 'rb'))
    elif args.cap_model == 'nic_plus':
        selected_cap_race_entries = pickle.load(open('bias_data/Woman-Snowboard/race_val_baselineft_cap_entries.pkl', 'rb'))


    if args.mask_race_words:
        race_words = ['white', 'caucasian','black', 'african', 'asian', 'latino', 'latina', 'latinx','hispanic', 'native', 'indigenous']
    else:
        race_words = []

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    ##################### ANN LIC score #######################
    if args.calc_ann_leak:
        print('--- calc ANN LIC score ---')
        ## Captioning ##
        if args.task == 'captioning':
            print('-- Task is Captioning --')
            d_train, d_test = make_train_test_split_race(args, race_val_obj_cap_entries)
            acc_list = []
            score_list = []
            light_acc_list, dark_acc_list = [], []
            rand_acc_list = []
            rand_score_list = []
            for caption_ind in range(5):
                trainANNCAPobject = BERT_ANN_leak_data(d_train, d_test, args, race_val_obj_cap_entries, race_words, tokenizer,
                                                args.max_seq_length, split='train', caption_ind=caption_ind)
                testANNCAPobject = BERT_ANN_leak_data(d_train, d_test, args, race_val_obj_cap_entries, race_words, tokenizer,
                                                args.max_seq_length, split='test', caption_ind=caption_ind)
                train_dataloader = torch.utils.data.DataLoader(trainANNCAPobject, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers)
                test_dataloader = torch.utils.data.DataLoader(testANNCAPobject, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
                # initialize gender classifier
                model = BERT_GenderClassifier(args, tokenizer)
                # calculate random predictions
                val_acc, val_loss, val_male_acc, val_female_acc, avg_score = calc_random_acc_score(args, model, test_dataloader)
                rand_acc_list.append(val_acc)
                rand_score_list.append(avg_score)
                # train and test
                val_acc, val_loss, val_light_acc, val_dark_acc, avg_score = calc_leak(args, model, train_dataloader, test_dataloader)
                acc_list.append(val_acc)
                light_acc_list.append(val_light_acc)
                dark_acc_list.append(val_dark_acc)
                score_list.append(avg_score)

            dark_avg_acc = sum(dark_acc_list) / len(dark_acc_list)
            light_avg_acc = sum(light_acc_list) / len(light_acc_list)
            avg_score = sum(score_list) / len(score_list)
            print('########### Reluts ##########')
            print(f"LIC score (LIC_D): {avg_score*100:.2f}%")
            #print(f'\t Light. Acc: {light_avg_acc*100:.2f}%')
            #print(f'\t Dark. Acc: {dark_avg_acc*100:.2f}%')
            print('#############################')


    ##################### MODEL LIC score #######################
    if args.calc_model_leak:
        print('--- calc MODEL LIC score ---')
        ## Captioning ##
        if args.task == 'captioning':
            print('-- Task is Captioning --')
            d_train, d_test = make_train_test_split_race(args, selected_cap_race_entries)
            trainMODELCAPobject = BERT_MODEL_leak_data(d_train, d_test, args, selected_cap_race_entries, race_words, tokenizer,
                                                args.max_seq_length, split='train')
            testMODELCAPobject = BERT_MODEL_leak_data(d_train, d_test, args, selected_cap_race_entries, race_words, tokenizer,
                                                args.max_seq_length, split='test')
            train_dataloader = torch.utils.data.DataLoader(trainMODELCAPobject, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers)
            test_dataloader = torch.utils.data.DataLoader(testMODELCAPobject, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
            # initialize gender classifier
            model = BERT_GenderClassifier(args, tokenizer)
            # calculate random predictions
            rand_val_acc, rand_val_loss, rand_val_light_acc, rand_val_dark_acc, rand_avg_score = calc_random_acc_score(args, model, test_dataloader)
            # train and test
            val_acc, val_loss, val_light_acc, val_dark_acc, avg_score = calc_leak(args, model, train_dataloader, test_dataloader)

            print('########### Reluts ##########')
            print(f'LIC score (LIC_M): {avg_score*100:.2f}%')
            #print(f'\t Light. Acc: {val_light_acc*100:.2f}%')
            #print(f'\t Dark. Acc: {val_dark_acc*100:.2f}%')
            print('#############################')


if __name__ == "__main__":
    parser = get_parser()
    args, unknown = parser.parse_known_args()
    print()
    print("---Start---")
    print('Seed:', args.seed)
    print("Epoch:", args.num_epochs)
    print("Freeze BERT:", args.freeze_bert)
    print("Learning rate:", args.learning_rate)
    print("Batch size:", args.batch_size)
    print("Calculate score:", args.calc_score)
    print("Task:", args.task)
    if args.task == 'captioning' and args.calc_model_leak:
        print("Captioning model:", args.cap_model)
    print("Gender or Race:", args.gender_or_race)
    print("Mask race words:", args.mask_race_words)

    if args.calc_ann_leak:
        print('Align vocab:', args.align_vocab)
        if args.align_vocab:
            print('Vocab of ', args.cap_model)

    print()

    main(args)

