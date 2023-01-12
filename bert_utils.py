import torch
import random
import nltk
nltk.download('punkt')

import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from operator import itemgetter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import AdamW


#Was previously in file bert_leakage.py
def make_train_test_split(args, gender_task_mw_entries):
    if args.balanced_data:
        male_entries, female_entries = [], []
        for entry in gender_task_mw_entries:
            if entry['bb_gender'] == 'Female':
                female_entries.append(entry)
            else:
                male_entries.append(entry)
        #print(len(female_entries))
        each_test_sample_num = round(len(female_entries) * args.test_ratio)
        each_train_sample_num = len(female_entries) - each_test_sample_num

        male_train_entries = [male_entries.pop(random.randrange(len(male_entries))) for _ in range(each_train_sample_num)]
        female_train_entries = [female_entries.pop(random.randrange(len(female_entries))) for _ in range(each_train_sample_num)]
        male_test_entries = [male_entries.pop(random.randrange(len(male_entries))) for _ in range(each_test_sample_num)]
        female_test_entries = [female_entries.pop(random.randrange(len(female_entries))) for _ in range(each_test_sample_num)]
        d_train = male_train_entries + female_train_entries
        d_test = male_test_entries + female_test_entries
        random.shuffle(d_train)
        random.shuffle(d_test)
        print('#train : #test = ', len(d_train), len(d_test))
    else:
        d_train, d_test = train_test_split(gender_task_mw_entries, test_size=args.test_ratio, random_state=args.seed,
                                   stratify=[entry['bb_gender'] for entry in gender_task_mw_entries])

    return d_train, d_test


def make_train_test_split_race(args, gender_task_race_entries):
    if args.balanced_data:
        light_entries, dark_entries = [], []
        for entry in gender_task_race_entries:
            if entry['bb_skin'] == 'Light':
                light_entries.append(entry)
            elif entry['bb_skin'] == 'Dark':
                dark_entries.append(entry)
        #print(len(female_entries))
        each_test_sample_num = round(len(dark_entries) * args.test_ratio)
        each_train_sample_num = len(dark_entries) - each_test_sample_num

        light_train_entries = [light_entries.pop(random.randrange(len(light_entries))) for _ in range(each_train_sample_num)]
        dark_train_entries = [dark_entries.pop(random.randrange(len(dark_entries))) for _ in range(each_train_sample_num)]
        light_test_entries = [light_entries.pop(random.randrange(len(light_entries))) for _ in range(each_test_sample_num)]
        dark_test_entries = [dark_entries.pop(random.randrange(len(dark_entries))) for _ in range(each_test_sample_num)]
        d_train = light_train_entries + dark_train_entries
        d_test = light_test_entries + dark_test_entries
        random.shuffle(d_train)
        random.shuffle(d_test)
        print(len(d_train), len(d_test))
    else:
        print('Balance data')
        #TODO: what? Why?

    return d_train, d_test


# Previously in bert_leakage.py, but they have a duplicate in race_bert_leakage 
def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum() / len(correct)
    return acc


# Previously in bert_leakage.py, but they have a duplicate in race_bert_leakage 
def calc_random_acc_score(args, model, test_dataloader):
    print("--- Random guess --")
    model = model.cuda()
    optimizer = None
    epoch = None
    val_loss, val_acc, val_male_acc, val_female_acc, avg_score = calc_leak_epoch_pass(args, test_dataloader, model, optimizer, False, print_every=500)

    return val_acc, val_loss, val_male_acc, val_female_acc, avg_score


def calc_leak(args, model, train_dataloader, test_dataloader):
    model = model.cuda()
    print("Num of Trainable Parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay = 1e-5)
    elif args.optimizer == 'adamw':
        param_optimizer = list(model.named_parameters())
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
        #no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            #{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, betas=(args.beta1, args.beta2), correct_bias=args.adam_correct_bias, eps=args.adam_epsilon)

    train_loss_arr = list()
    train_acc_arr = list()

    # training
    for epoch in range(args.num_epochs):
        # train
        train_loss, train_acc, _, _, _ = calc_leak_epoch_pass(args, train_dataloader, model, optimizer, True, print_every=500)
        train_loss_arr.append(train_loss)
        train_acc_arr.append(train_acc)
        if epoch % 5 == 0:
            print('train, {0}, train loss: {1:.2f}, train acc: {2:.2f}'.format(epoch, \
                train_loss*100, train_acc*100))

    print("Finish training")
    print('{0}: train acc: {1:2f}'.format(epoch, train_acc))

    
    # validation
    if args.calc_mw_acc:
        val_loss, val_acc, val_male_acc, val_female_acc, avg_score = calc_leak_epoch_pass(args, test_dataloader, model, optimizer, False, print_every=500)
        print('val, {0}, val loss: {1:.2f}, val acc: {2:.2f}'.format(epoch, val_loss*100, val_acc *100))
        
        print('val, {0}, val loss: {1:.2f}, Male val acc: {2:.2f}'.format(epoch, val_loss*100, val_male_acc *100))
        print('val, {0}, val loss: {1:.2f}, Female val acc: {2:.2f}'.format(epoch, val_loss*100, val_female_acc *100))

        return val_acc, val_loss, val_male_acc, val_female_acc, avg_score
    
    elif args.calc_race_acc:
        val_loss, val_acc, val_light_acc, val_dark_acc, avg_score = calc_leak_epoch_pass(epoch, test_dataloader, model, optimizer, False, print_every=500)
        print('val, {0}, val loss: {1:.2f}, val acc: {2:.2f}'.format(epoch, val_loss*100, val_acc *100))
        
        print('val, {0}, val loss: {1:.2f}, Light val acc: {2:.2f}'.format(epoch, val_loss*100, val_light_acc *100))
        print('val, {0}, val loss: {1:.2f}, Dark val acc: {2:.2f}'.format(epoch, val_loss*100, val_dark_acc *100))

        return val_acc, val_loss, val_light_acc, val_dark_acc, avg_score
    
    
    
    #TODO: epoch is an unused variable in this function, I removed it and changed to args.
# TODO: I would also maybe rename this to smth else since its the training or validation loop

def calc_leak_epoch_pass(args, data_loader, model, optimizer, training, print_every):
    t_loss = 0.0
    n_processed = 0
    preds = list()
    truth = list()
    
    if args.gender_or_race == 'gender':
        male_preds_all, female_preds_all = list(), list()
        male_truth_all, female_truth_all = list(), list()

        if training:
            model.train()
        else:
            model.eval()

        if args.store_topk_gender_pred:
            all_male_pred_values, all_female_pred_values = [], []
            all_male_inputs, all_female_inputs = [], []

        total_score = 0 # for calculate scores

        cnt_data = 0
        for ind, (input_ids, attention_mask, token_type_ids, gender_target, img_id) in tqdm(enumerate(data_loader), leave=False): # images are not provided
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            token_type_ids = token_type_ids.cuda()

            gender_target = torch.squeeze(gender_target).cuda()
            predictions = model(input_ids, attention_mask, token_type_ids)
            cnt_data += predictions.size(0)

            loss = F.cross_entropy(predictions, gender_target, reduction='mean')

            if not training and args.store_topk_gender_pred:
                pred_values = np.amax(F.softmax(predictions, dim=1).cpu().detach().numpy(), axis=1).tolist()
                pred_genders = np.argmax(F.softmax(predictions, dim=1).cpu().detach().numpy(), axis=1)

                for pv, pg, imid, ids in zip(pred_values, pred_genders, img_id, input_ids):
                    tokens = model.tokenizer.convert_ids_to_tokens(ids)
                    text = model.tokenizer.convert_tokens_to_string(tokens)
                    text = text.replace('[PAD]', '')
                    if pg == 0:
                        all_male_pred_values.append(pv)
                        all_male_inputs.append({'img_id': imid, 'text': text})
                    else:
                        all_female_pred_values.append(pv)
                        all_female_inputs.append({'img_id': imid, 'text': text})

            if not training and args.calc_score:
                pred_genders = np.argmax(F.softmax(predictions, dim=1).cpu().detach(), axis=1)
                gender_target = gender_target.cpu().detach()
                correct = torch.eq(pred_genders, gender_target)
                #if ind == 0:
                #    print('correct:', correct, correct.shape)

                pred_score_tensor = torch.zeros_like(correct, dtype=float)
                for i in range(pred_score_tensor.size(0)):
                    male_score = F.softmax(predictions, dim=1).cpu().detach()[i,0]
                    female_score = F.softmax(predictions, dim=1).cpu().detach()[i,1]

                    if male_score >= female_score:
                        pred_score = male_score
                    else:
                        pred_score = female_score

                    pred_score_tensor[i] = pred_score

                scores_tensor = correct.int() * pred_score_tensor
                correct_score_sum = torch.sum(scores_tensor)
                total_score += correct_score_sum.item()

            predictions = np.argmax(F.softmax(predictions, dim=1).cpu().detach().numpy(), axis=1)
            preds += predictions.tolist()
            truth += gender_target.cpu().numpy().tolist()

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            t_loss += loss.item()
            n_processed += len(gender_target)

            if (ind + 1) % print_every == 0 and training:
                print('{0}: task loss: {1:4f}'.format(ind + 1, t_loss / n_processed))

            if args.calc_mw_acc and not training:
                male_target_ind = [i for i, x in enumerate(gender_target.cpu().numpy().tolist()) if x == 0]
                female_target_ind = [i for i, x in enumerate(gender_target.cpu().numpy().tolist()) if x == 1]
                male_pred = [*itemgetter(*male_target_ind)(predictions.tolist())]
                female_pred = [*itemgetter(*female_target_ind)(predictions.tolist())]
                male_target = [*itemgetter(*male_target_ind)(gender_target.cpu().numpy().tolist())]
                female_target = [*itemgetter(*female_target_ind)(gender_target.cpu().numpy().tolist())]
                male_preds_all += male_pred
                male_truth_all += male_target
                female_preds_all += female_pred
                female_truth_all += female_target

        acc = accuracy_score(truth, preds)

        if args.calc_mw_acc and not training:
            male_acc = accuracy_score(male_truth_all, male_preds_all)
            female_acc = accuracy_score(female_truth_all, female_preds_all)
        else:
            male_acc, female_acc = None, None

        return t_loss / n_processed, acc, male_acc, female_acc, total_score / cnt_data




    else:
        light_preds_all, dark_preds_all = list(), list()
    light_truth_all, dark_truth_all = list(), list()

    if training:
        model.train()
    else:
        model.eval()

    if args.store_topk_race_pred:
        all_light_pred_values, all_dark_pred_values = [], []
        all_light_inputs, all_dark_inputs = [], []

    total_score = 0 # for calculate scores

    cnt_data = 0
    for ind, (input_ids, attention_mask, token_type_ids, race_target, img_id) in tqdm(enumerate(data_loader), leave=False): # images are not provided
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()
        token_type_ids = token_type_ids.cuda()

        race_target = torch.squeeze(race_target).cuda()
        #if ind == 0:
        #    for j in range(30):
        #        print(model.tokenizer.convert_ids_to_tokens(input_ids[j]))
        #    print(input_vec.shape) #[batch, num_obj]
        #    print(gender_target.shape) #[batch, 1] or [batch]
        predictions = model(input_ids, attention_mask, token_type_ids)
        #if ind == 0:
        #    print(predictions.shape) #[batch, 2]
        #    print(predictions)
        #    print(F.softmax(predictions, dim=1).cpu().detach().numpy())
        cnt_data += predictions.size(0)

        loss = F.cross_entropy(predictions, race_target, reduction='mean')

        if not training and args.store_topk_race_pred:
            pred_values = np.amax(F.softmax(predictions, dim=1).cpu().detach().numpy(), axis=1).tolist()
            pred_races = np.argmax(F.softmax(predictions, dim=1).cpu().detach().numpy(), axis=1)

            for pv, pr, imid, ids in zip(pred_values, pred_races, img_id, input_ids):
                tokens = model.tokenizer.convert_ids_to_tokens(ids)
                text = model.tokenizer.convert_tokens_to_string(tokens)
                text = text.replace('[PAD]', '')
                if pr == 0:
                    all_light_pred_values.append(pv)
                    all_light_inputs.append({'img_id': imid, 'text': text})
                else:
                    all_dark_pred_values.append(pv)
                    all_dark_inputs.append({'img_id': imid, 'text': text})

        if not training and args.calc_score:
            pred_races = np.argmax(F.softmax(predictions, dim=1).cpu().detach(), axis=1)
            race_target = race_target.cpu().detach()
            correct = torch.eq(pred_races, race_target)
            #if ind == 0:
            #    print('correct:', correct, correct.shape)

            pred_score_tensor = torch.zeros_like(correct, dtype=float)
            for i in range(pred_score_tensor.size(0)):
                #if ind == 0:
                #    print(F.softmax(predictions, dim=1).cpu().detach().shape)
                light_score = F.softmax(predictions, dim=1).cpu().detach()[i,0]
                dark_score = F.softmax(predictions, dim=1).cpu().detach()[i,1]
                if light_score >= dark_score:
                    pred_score = light_score
                else:
                    pred_score = dark_score
                #if ind == 0:
                #    print(male_score)
                #    print(female_score)
                pred_score_tensor[i] = pred_score
            #if ind == 0:
            #    print('pred_score_tensor:',pred_score_tensor, pred_score_tensor.shape)

            scores_tensor = correct.int() * pred_score_tensor
            correct_score_sum = torch.sum(scores_tensor)
            total_score += correct_score_sum.item()
             

        predictions = np.argmax(F.softmax(predictions, dim=1).cpu().detach().numpy(), axis=1)
        #if ind == 0 and epoch % 30 == 0:
        #    print(predictions)
        preds += predictions.tolist()
        truth += race_target.cpu().numpy().tolist()
        #if ind == 0 and epoch % 30 == 0:
        #    print('preds:', preds)
        #    print('truth:', truth)

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        t_loss += loss.item()
        n_processed += len(race_target)

        if (ind + 1) % print_every == 0 and training:
            print('{0}: task loss: {1:4f}'.format(ind + 1, t_loss / n_processed))

        if args.calc_race_acc and not training:
            light_target_ind = [i for i, x in enumerate(race_target.cpu().numpy().tolist()) if x == 0]
            dark_target_ind = [i for i, x in enumerate(race_target.cpu().numpy().tolist()) if x == 1]
            light_pred = [*itemgetter(*light_target_ind)(predictions.tolist())]
            dark_pred = [*itemgetter(*dark_target_ind)(predictions.tolist())]
            light_target = [*itemgetter(*light_target_ind)(race_target.cpu().numpy().tolist())]
            dark_target = [*itemgetter(*dark_target_ind)(race_target.cpu().numpy().tolist())]
            light_preds_all += light_pred
            light_truth_all += light_target
            dark_preds_all += dark_pred
            dark_truth_all += dark_target

    acc = accuracy_score(truth, preds)

    if args.calc_race_acc and not training:
        light_acc = accuracy_score(light_truth_all, light_preds_all)
        dark_acc = accuracy_score(dark_truth_all, dark_preds_all)
    else:
        light_acc, dark_acc = None, None

    if args.store_topk_race_pred and not training:
        all_light_pred_values = np.array(all_light_pred_values)
        all_dark_pred_values = np.array(all_dark_pred_values)
        #Light
        light_ind = all_light_pred_values.argsort()[-args.topk_race_pred:][::-1]
        light_topk_inputs = np.array(all_light_inputs)[light_ind]
        light_topk_scores = all_light_pred_values[light_ind]
        print("topk inputs (Light)")
        print(light_topk_inputs)
        print(light_topk_scores)
        print(light_ind)
        print()
        #Dark
        dark_ind = all_dark_pred_values.argsort()[-args.topk_race_pred:][::-1]
        dark_topk_inputs = np.array(all_dark_inputs)[dark_ind]
        dark_topk_scores = all_dark_pred_values[dark_ind]
        print("topk inputs (Dark)")
        print(dark_topk_inputs)
        print(dark_topk_scores)
        print(dark_ind)

    #if args.calc_score and not training:
    #    print("### AVG SCORE ###")
    #    print(total_score / (args.batch_size * len(data_loader)))
    #    print("#################")

    ###return t_loss / n_processed, acc, light_acc, dark_acc, total_score / (args.batch_size * len(data_loader))
    return t_loss / n_processed, acc, light_acc, dark_acc, total_score / cnt_data