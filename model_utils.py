import torch
from transformers import AdamW
from utils import modeling
import numpy as np
import pandas as pd


def model_setup(num_labels, model_select, device, config, gen, dropout, dropoutrest):
    print("current dropout is: ", dropout, dropoutrest)
    if model_select == 'Bert':
        print(100 * "#")
        print("using Bert")
        print(100 * "#")
        model = modeling.bert_classifier(num_labels, model_select, gen, dropout, dropoutrest).to(device)
        for n, p in model.named_parameters():
            if "bert.embeddings" in n:
                p.requires_grad = False
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if n.startswith('bert.encoder')],
             'lr': float(config['bert_lr'])},
            {'params': [p for n, p in model.named_parameters() if n.startswith('linear')],
             'lr': float(config['fc_lr'])},
            {'params': [p for n, p in model.named_parameters() if n.startswith('out')], 'lr': float(config['fc_lr'])}
        ]
    elif model_select == 'bart-base':
        print(100 * "#")
        print("using Bart")
        print(100 * "#")
        model = modeling.bart_classifier(num_labels, model_select, gen, dropout, dropoutrest).to(device)
        for n, p in model.named_parameters():
            if "bart.shared.weight" in n or "bart.encoder.embed" in n:
                p.requires_grad = False
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if n.startswith('bart.encoder.layer')],
             'lr': float(config['bert_lr'])},
            {'params': [p for n, p in model.named_parameters() if n.startswith('linear')],
             'lr': float(config['fc_lr'])},
            {'params': [p for n, p in model.named_parameters() if n.startswith('out')], 'lr': float(config['fc_lr'])}
        ]

    optimizer = AdamW(optimizer_grouped_parameters)

    return model, optimizer


def model_preds(loader, model, device, loss_function, cl_loss_function):
    preds = []
    valtest_loss = []
    for b_id, sample_batch in enumerate(loader):
        dict_batch = batch_fn(sample_batch)
        inputs = {k: v.to(device) for k, v in dict_batch.items()}
        outputs, features = model(**inputs)
        preds.append(outputs)

        loss = loss_function(outputs, inputs['gt_label']) + cl_loss_function(features, inputs['gt_label'])
        valtest_loss.append(loss.item())

    return torch.cat(preds, 0), valtest_loss


def test_preds(loader, model, device, loss_function, cl_loss_function, text, target, there_label):
    preds = []
    here_label = []
    for b_id, sample_batch in enumerate(loader):
        dict_batch = batch_fn(sample_batch)
        inputs = {k: v.to(device) for k, v in dict_batch.items()}
        outputs, features = model(**inputs)
        p = outputs.argmax(axis=1)
        p = p.detach().cpu().numpy()
        preds = np.concatenate((preds, p), 0)
        # l= inputs['gt_label'].detach().cpu().numpy()
        # here_label= np.concatenate((here_label, l), 0)
        # preds.append(p)

    # preds=preds.detach().cpu().numpy()

    # loss = loss_function(outputs, inputs['gt_label']) + cl_loss_function(features, inputs['gt_label'])
    # valtest_loss.append(loss.item())
    test = pd.DataFrame({'text': text, 'topic': target, 'label': there_label, 'pred': preds, })
    test.to_csv('save_ba.csv', index=False, sep=',')
    # return torch.cat(preds, 0), valtest_loss
    return preds, text, target


def batch_fn(sample_batch):
    dict_batch = {}
    dict_batch['input_ids'] = sample_batch[0]
    dict_batch['attention_mask'] = sample_batch[1]
    dict_batch['gt_label'] = sample_batch[2]
    # if len(sample_batch) > 3:
    #     dict_batch['token_type_ids'] = sample_batch[-2]

    return dict_batch