import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import pandas as pd
import argparse
import json
import os
import gc
import gspread
import utils.preprocessing as pp
import utils.data_helper as dh
from transformers import AdamW
from utils import modeling, evaluation, model_utils
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, classification_report

from torch.utils.tensorboard import SummaryWriter
from pytorchtools import EarlyStopping

# import loss

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class unitedCLLoss(nn.Module):
    def __init__(self, temperature, contrast_mode='all'):
        super(unitedCLLoss, self).__init__()

        self.temperature = temperature
        self.contrast_mode = contrast_mode

    def forward(self, features, labels, mask=None):
        """
            Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
            It also supports the unsupervised contrastive loss in SimCLR
        """
        """ Compute loss for model. If both `labels` and `mask` are None,
            it degenerates to SimCLR unsupervised loss:
            https://arxiv.org/pdf/2002.05709.pdf
            Args:
                features: hidden vector of shape [bsz, n_views, ...].
                labels: ground truth of shape [bsz].
                mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                    has the same class as sample i. Can be asymmetric.
            Returns:
                A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                labels = torch.cat([labels, labels], dim=0)

            mask = torch.eq(labels, labels.T).float().add(0.0000001).to(
                device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask_pos = mask * logits_mask
        mask_neg = (torch.ones_like(mask) - mask) * logits_mask

        similarity = torch.exp(torch.mm(anchor_feature, contrast_feature.t()) / self.temperature)

        pos = torch.sum(similarity * mask_pos, 1)
        neg = torch.sum(similarity * mask_neg, 1)
        loss = -(torch.mean(torch.log(pos / (pos + neg))))

        return loss


def compute_performance(preds, y, trainvaltest, step, args, seed):
    print("preds:", preds, preds.size())
    print("y:", y, y.size())
    preds_np = preds.cpu().numpy()
    preds_np = np.argmax(preds_np, axis=1)
    y_train2_np = y.cpu().numpy()
    results_weighted = precision_recall_fscore_support(y_train2_np, preds_np, average='macro')

    print("-------------------------------------------------------------------------------------")
    print(trainvaltest + " classification_report for step: {}".format(step))
    target_names = ['Against', 'Favor', 'Neutral']
    print(classification_report(y_train2_np, preds_np, target_names=target_names, digits=4))
    ###############################################################################################
    ################            Precision, recall, F1 to csv                     ##################
    ###############################################################################################
    # y_true = out_label_ids
    # y_pred = preds
    results_twoClass = precision_recall_fscore_support(y_train2_np, preds_np, average=None)  # 包含了每个类别的精确度、召回率、F1分数和支持数。
    results_weighted = precision_recall_fscore_support(y_train2_np, preds_np,
                                                       average='macro')  # 包含了宏平均的精确度、召回率、F1分数和支持数。
    print("results_weighted:", results_weighted)
    result_overall = [results_weighted[0], results_weighted[1], results_weighted[2]]
    result_against = [results_twoClass[0][0], results_twoClass[1][0], results_twoClass[2][0]]
    result_favor = [results_twoClass[0][1], results_twoClass[1][1], results_twoClass[2][1]]
    result_neutral = [results_twoClass[0][2], results_twoClass[1][2], results_twoClass[2][2]]

    print("result_overall:", result_overall)
    print("result_favor:", result_favor)
    print("result_against:", result_against)
    print("result_neutral:", result_neutral)

    result_id = ['train', args['gen'], step, seed, args['dropout'], args['dropoutrest']]
    result_one_sample = result_id + result_against + result_favor + result_neutral + result_overall
    result_one_sample = [result_one_sample]
    print("result_one_sample:", result_one_sample)

    # if results_weighted[2]>best_train_f1macro:
    #     best_train_f1macro = results_weighted[2]
    #     best_train_result = result_one_sample

    results_df = pd.DataFrame(result_one_sample)
    print("results_df are:", results_df.head())
    results_df.to_csv('./results_' + trainvaltest + '_df.csv', index=False, mode='a', header=False)
    print('./results_' + trainvaltest + '_df.csv save, done!')
    print("----------------------------------------------------------------------------")

    return results_weighted[2], result_one_sample


def run_classifier():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', help='Name of the cofig data file', default='../config/config-bert.txt',
                        required=False)
    parser.add_argument('-g', '--gen', help='Generation number of student model', default=1, required=False)
    parser.add_argument('-s', '--seed', help='Random seed', default=0, required=False)
    parser.add_argument('-d', '--dropout', help='Dropout rate', default=0.1, required=False)
    parser.add_argument('-d2', '--dropoutrest', help='Dropout rate for rest generations', default=0.25, required=False)
    parser.add_argument('-train', '--train_data', help='Name of the training data file',
                        default='../data/vast_train.csv', required=False)
    parser.add_argument('-dev', '--dev_data', help='Name of the dev data file',
                        default='../data/vast_dev.csv', required=False)
    parser.add_argument('-test', '--test_data', help='Name of the test data file',
                        default='../data/vast_test.csv', required=False)

    # parser.add_argument('-kg', '--kg_data', help='Name of the kg test data file',
    #                     default='../data/text_entailment/HC_train_kg.csv', required=False)
    parser.add_argument('-clipgrad', '--clipgradient', type=str, default='True',
                        help='whether clip gradient when over 2', required=False)
    parser.add_argument('-step', '--savestep', type=int, default=3, help='whether clip gradient when over 2',
                        required=False)
    parser.add_argument('-p', '--percent', type=int, default=100, help='whether clip gradient when over 2',
                        required=False)
    parser.add_argument('-es_step', '--earlystopping_step', type=int, default=5,
                        help='whether clip gradient when over 2', required=False)

    args = vars(parser.parse_args())

    sheet_num = 4  # Google sheet number
    num_labels = 3  # Favor, Against and None
    #     random_seeds = [0,1,2,3,4,42]
    random_seeds = []
    random_seeds.append(int(args['seed']))

    # create normalization dictionary for preprocessing
    with open("./noslang_data.json", "r") as f:
        data1 = json.load(f)
    data2 = {}
    with open("./emnlp_dict.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            row = line.split('\t')
            data2[row[0]] = row[1].rstrip()
    norm_dict = {**data1, **data2}

    # load config file
    with open(args['config_file'], 'r') as f:
        config = dict()
        for l in f.readlines():
            config[l.strip().split(":")[0]] = l.strip().split(":")[1]
    model_select = config['model_select']

    # Use GPU or not
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = "cpu"

    best_result, best_against, best_favor, best_val, best_val_against, best_val_favor, = [], [], [], [], [], []
    for seed in random_seeds:
        print("current random seed: ", seed)
        # a=str(args['dropout'])
        # b=str(args['dropoutrest'])
        # log_dir = os.path.join('./tensorboard/tensorboard_train'+str(args['percent'])+'_d0'+str(args['dropout'])+'_d1'+str(args['dropoutrest']+'_seed'+str(seed)+'_gen'+str(args['gen'])), 'train')
        log_dir = os.path.join(
            './tensorboard/tensorboard_train' + str(args['percent']) + '_d0' + str(args['dropout']) + '_d1' + str(
                args['dropoutrest']) + '_seed' + str(seed) + '_gen' + str(args['gen']), 'train')
        train_writer = SummaryWriter(log_dir=log_dir)

        log_dir = os.path.join(
            './tensorboard/tensorboard_train' + str(args['percent']) + '_d0' + str(args['dropout']) + '_d1' + str(
                args['dropoutrest']) + '_seed' + str(seed) + '_gen' + str(args['gen']), 'val')
        val_writer = SummaryWriter(log_dir=log_dir)

        log_dir = os.path.join(
            './tensorboard/tensorboard_train' + str(args['percent']) + '_d0' + str(args['dropout']) + '_d1' + str(
                args['dropoutrest']) + '_seed' + str(seed) + '_gen' + str(args['gen']), 'test')
        test_writer = SummaryWriter(log_dir=log_dir)

        # set up the random seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        text_test, target_test, label_test = pp.clean_test(args['test_data'])
        x_train, y_train, x_train_target = pp.clean_all(args['train_data'], norm_dict)

        x_test, y_test, x_test_target = pp.clean_all(args['test_data'], norm_dict)
        x_val, y_val, x_val_target = pp.clean_all(args['dev_data'], norm_dict)
        # x_test_kg, y_test_kg, x_test_target_kg = pp.clean_all(args['kg_data'], norm_dict)
        x_train_all = [x_train, y_train, x_train_target]
        x_val_all = [x_val, y_val, x_val_target]
        x_test_all = [x_test, y_test, x_test_target]
        # x_test_kg_all = [x_test_kg,y_test_kg,x_test_target_kg]
        # if int(args['gen']) >= 1:
        #     print("Current generation is: ", args['gen'])
        #     x_train_all = [a+b for a,b in zip(x_train_all, x_test_kg_all)]
        print(x_test_all[0][0], x_test_all[1][0], x_test_all[2][0])

        # prepare for model
        loader, gt_label = dh.data_helper_bert(x_train_all, x_val_all, x_test_all, model_select, config)
        trainloader, valloader, testloader, trainloader2 = loader[0], loader[1], loader[2], loader[3]
        y_train, y_val, y_test, y_train2 = gt_label[0], gt_label[1], gt_label[2], gt_label[3]
        y_val, y_test, y_train2 = y_val.to(device), y_test.to(device), y_train2.to(device)

        # train setup
        model, optimizer = model_utils.model_setup(num_labels, model_select, device, config, int(args['gen']),
                                                   float(args['dropout']), float(args['dropoutrest']))
        loss_function = nn.CrossEntropyLoss()
        cl_loss_cunction = unitedCLLoss(0.14)
        sum_loss = []
        val_f1_average, val_f1_against, val_f1_favor = [], [], []
        test_f1_average, test_f1_against, test_f1_favor, test_kg = [], [], [], []

        # early stopping

        es_intermediate_step = len(trainloader) // args['savestep']
        patience = args['earlystopping_step']  # the number of iterations that loss does not further decrease
        # patience = es_intermediate_step   # the number of iterations that loss does not further decrease
        early_stopping = EarlyStopping(patience, verbose=True)
        print(100 * "#")
        # print("len(trainloader):",len(trainloader))
        # print("args['savestep']:",args['savestep'])
        print("early stopping occurs when the loss does not decrease after {} steps.".format(patience))
        print(100 * "#")
        # print(bk)
        # init best val/test results
        best_train_f1macro = 0
        best_train_result = []
        best_val_f1macro = 0
        best_val_result = []
        best_test_f1macro = 0
        best_test_result = []

        best_val_loss = 100000
        best_val_loss_result = []
        best_test_loss = 100000
        best_test_loss_result = []
        # start training
        print(100 * "#")
        print("clipgradient:", args['clipgradient'] == 'True')
        print(100 * "#")
        step = 0
        # start training
        for epoch in range(0, int(config['total_epochs'])):
            print('Epoch:', epoch)
            train_loss = []
            model.train()
            for b_id, sample_batch in enumerate(trainloader):
                model.train()
                optimizer.zero_grad()
                dict_batch = model_utils.batch_fn(sample_batch)
                inputs = {k: v.to(device) for k, v in dict_batch.items()}
                outputs, features = model(**inputs)
                cl_loss = cl_loss_cunction(features, inputs['gt_label'])
                sd_loss = loss_function(outputs, inputs['gt_label'])

                # cl_loss=cl_loss_cunction(features,inputs['gt_label'])
                loss = sd_loss + cl_loss
                loss.backward()

                # nn.utils.clip_grad_norm_(model.parameters(), 2)
                if args['clipgradient'] == 'True':
                    nn.utils.clip_grad_norm_(model.parameters(), 2)

                optimizer.step()
                step += 1
                train_loss.append(loss.item())

                split_step = len(trainloader) // args['savestep']
                if step % split_step == 0:
                    model.eval()
                    with torch.no_grad():
                        preds_train, loss_train_inval_mode = model_utils.model_preds(trainloader2, model, device,
                                                                                     loss_function,
                                                                                     cl_loss_cunction)  # train2和train数据一样
                        preds_val, loss_val = model_utils.model_preds(valloader, model, device, loss_function,
                                                                      cl_loss_cunction)
                        preds_test, loss_test = model_utils.model_preds(testloader, model, device, loss_function,
                                                                        cl_loss_cunction)
                        pr_test, te_test, ta_test = model_utils.test_preds(testloader, model, device, loss_function,
                                                                           cl_loss_cunction, text_test, target_test,
                                                                           label_test)
                        print(100 * "#")
                        print("at step: {}".format(step))
                        print("train_loss", train_loss, len(train_loss), sum(train_loss) / len(train_loss))
                        print("loss_val", loss_val, len(loss_val), sum(loss_val) / len(loss_val))
                        print("loss_test", loss_test, len(loss_test), sum(loss_test) / len(loss_test))

                        # print(bk)

                        train_writer.add_scalar('loss', sum(train_loss) / len(train_loss),
                                                step)  # 将训练损失的平均值记录到TensorBoard中
                        val_writer.add_scalar('loss', sum(loss_val) / len(loss_val), step)
                        test_writer.add_scalar('loss', sum(loss_test) / len(loss_test), step)

                        f1macro_train, result_one_sample_train = compute_performance(preds_train, y_train2, 'training',
                                                                                     step, args, seed)
                        f1macro_val, result_one_sample_val = compute_performance(preds_val, y_val, 'validation', step,
                                                                                 args, seed)
                        f1macro_test, result_one_sample_test = compute_performance(preds_test, y_test, 'test', step,
                                                                                   args, seed)

                        train_writer.add_scalar('f1macro', f1macro_train, step)
                        val_writer.add_scalar('f1macro', f1macro_val, step)
                        test_writer.add_scalar('f1macro', f1macro_test, step)

                        if f1macro_val > best_val_f1macro:
                            best_val_f1macro = f1macro_val
                            best_val_result = result_one_sample_val
                            print(100 * "#")
                            print(
                                "best f1-macro validation updated at epoch :{}, to: {}".format(epoch, best_val_f1macro))
                            best_test_f1macro = f1macro_test
                            best_test_result = result_one_sample_test
                            print("best f1-macro test updated at epoch :{}, to: {}".format(epoch, best_test_f1macro))
                            print(100 * "#")

                        avg_val_loss = sum(loss_val) / len(loss_val)
                        avg_test_loss = sum(loss_test) / len(loss_test)
                        if avg_val_loss < best_val_loss:
                            best_val_loss = avg_val_loss
                            best_val_loss_result = result_one_sample_val
                            print(100 * "#")
                            print("best loss validation updated at epoch :{}, to: {}".format(epoch, best_val_loss))
                            best_test_loss = avg_test_loss
                            best_test_loss_result = result_one_sample_test
                            print("best loss test updated at epoch :{}, to: {}".format(epoch, best_test_loss))
                            print(100 * "#")

                        _, f1_average, f1_against, f1_favor = evaluation.compute_f1(preds_val, y_val)
                        val_f1_against.append(f1_against)
                        val_f1_favor.append(f1_favor)
                        val_f1_average.append(f1_average)
                        _, f1_average, f1_against, f1_favor = evaluation.compute_f1(preds_test, y_test)

                        test_f1_against.append(f1_against)
                        test_f1_favor.append(f1_favor)
                        test_f1_average.append(f1_average)

                        # early stopping
                        print("loss_val:", loss_val, "average is: ", sum(loss_val) / len(loss_val))
                        early_stopping(sum(loss_val) / len(loss_val), model)
                        if early_stopping.early_stop:
                            print(100 * "!")
                            print("Early stopping occurs at step: {}, stop training.".format(step))
                            print(100 * "!")
                            break
                    model.train()

            if early_stopping.early_stop:
                print(100 * "!")
                print("Early stopping, training ends")
                print(100 * "!")
                break

            sum_loss.append(sum(train_loss) / len(train_loss))
            print(sum_loss[epoch])

        best_val_result[0][0] = 'best validation'
        results_df = pd.DataFrame(best_val_result)
        print("results_df are:", results_df.head())
        results_df.to_csv('./results_validation_df.csv', index=False, mode='a', header=False)
        print('./results_validation_df.csv save, done!')
        ###
        results_df = pd.DataFrame(best_val_result)
        print("results_df are:", results_df.head())
        results_df.to_csv('./best_results_validation_df.csv', index=False, mode='a', header=False)
        print('./best_results_validation_df.csv save, done!')
        ###
        best_val_loss_result[0][0] = 'best validation'
        results_df = pd.DataFrame(best_val_loss_result)
        print("results_df are:", results_df.head())
        results_df.to_csv('./best_loss_results_validation_df.csv', index=False, mode='a', header=False)
        print('./best_loss_results_validation_df.csv save, done!')
        #########################################################
        best_test_result[0][0] = 'best test'
        results_df = pd.DataFrame(best_test_result)
        print("results_df are:", results_df.head())
        results_df.to_csv('./results_test_df.csv', index=False, mode='a', header=False)
        print('./results_test_df.csv save, done!')
        ###
        results_df = pd.DataFrame(best_test_result)
        print("results_df are:", results_df.head())
        results_df.to_csv('./best_results_test_df.csv', index=False, mode='a', header=False)
        print('./best_results_test_df.csv save, done!')
        ###
        best_test_loss_result[0][0] = 'best test'
        results_df = pd.DataFrame(best_test_loss_result)
        print("results_df are:", results_df.head())
        results_df.to_csv('./best_loss_results_test_df.csv', index=False, mode='a', header=False)
        print('./best_loss_results_test_df.csv save, done!')
        #########################################################
        # model that performs best on the dev set is evaluated on the test set
        best_epoch = [index for index, v in enumerate(val_f1_average) if v == max(val_f1_average)][-1]
        best_against.append(test_f1_against[best_epoch])
        best_favor.append(test_f1_favor[best_epoch])
        best_result.append(test_f1_average[best_epoch])

        print("******************************************")
        print("dev results with seed {} on all epochs".format(seed))
        print(val_f1_average)
        best_val_against.append(val_f1_against[best_epoch])
        best_val_favor.append(val_f1_favor[best_epoch])
        best_val.append(val_f1_average[best_epoch])
        print("******************************************")
        print("test results with seed {} on all epochs".format(seed))
        print(test_f1_average)
        print("******************************************")
        print(max(best_result))
        print(best_result)
    # save to Google sheet
    save_result = []
    save_result.append(best_against)
    save_result.append(best_favor)
    save_result.append(best_result)  # results on test set
    save_result.append(best_val_against)
    save_result.append(best_val_favor)
    save_result.append(best_val)
    print(save_result)  # results on val set

if __name__ == "__main__":
    run_classifier()
