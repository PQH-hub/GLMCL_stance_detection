import torch
import transformers
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, AutoTokenizer, BertweetTokenizer, BartTokenizer

transformers.logging.set_verbosity_error()


# Tokenization
def convert_data_to_ids(tokenizer, target, text, label, config):
    # 存储目标数据和文本数据
    concat_sent = []
    for tar, sent in zip(target, text):
        concat_sent.append([' '.join(sent), ' '.join(tar)])
    encoded_dict = tokenizer.batch_encode_plus(
        concat_sent,
        add_special_tokens=True,  # 在序列的开头和结尾添加特殊标记'[CLS]' and '[SEP]'
        max_length=int(config['max_tok_len']),  # 指定最大的序列长度，超过该长度的部分将被截断，不足的部分将被填充
        padding='max_length',  # 指定填充方式为填充到最大长度
        return_attention_mask=True,  # 返回注意力掩码，用于指示模型应该关注输入序列的哪些部分
        truncation=True, # 如果序列长度超过 max_length，则进行截断
    )
    encoded_dict['gt_label'] = label  # 将标签数据 label 存储在 encoded_dict 字典中的 'gt_label' 键下，用于后续训练或评估时的参考

    return encoded_dict   # 返回 encoded_dict 字典，其中包含转换后的ids表示和标签数据


# BERT/BERTweet tokenizer
def data_helper_bert(x_train_all, x_val_all, x_test_all, model_select, config):
    print('Loading data')
    x_train, y_train, x_train_target = x_train_all[0], x_train_all[1], x_train_all[2]
    x_val, y_val, x_val_target = x_val_all[0], x_val_all[1], x_val_all[2]
    x_test, y_test, x_test_target = x_test_all[0], x_test_all[1], x_test_all[2]
    print("Length of original x_train: %d" % (len(x_train)))
    print("Length of original x_val: %d, the sum is: %d" % (len(x_val), sum(y_val)))
    print("Length of original x_test: %d, the sum is: %d" % (len(x_test), sum(y_test)))

    # 加载分词器
    if model_select == 'Bertweet':
        tokenizer = BertweetTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)
    elif model_select == 'bart-base':
        tokenizer = BartTokenizer.from_pretrained(
            "C:\\Users\\20171\\Desktop\\Models\\bart-base", normalization=True)
    elif model_select == 'Bert':
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    # tokenization
    train_encoded_dict = convert_data_to_ids(tokenizer, x_train_target, x_train, y_train, config)
    val_encoded_dict = convert_data_to_ids(tokenizer, x_val_target, x_val, y_val, config)
    test_encoded_dict = convert_data_to_ids(tokenizer, x_test_target, x_test, y_test, config)

    trainloader, y_train = data_loader(train_encoded_dict, int(config['batch_size']), model_select, 'train')
    valloader, y_val = data_loader(val_encoded_dict, int(config['batch_size']), model_select, 'val')
    testloader, y_test = data_loader(test_encoded_dict, int(config['batch_size']), model_select, 'test')
    trainloader2, y_train2 = data_loader(train_encoded_dict, int(config['batch_size']), model_select, 'train2')

    print("Length of final x_train: %d" % (len(y_train)))

    return (trainloader, valloader, testloader, trainloader2), (y_train, y_val, y_test, y_train2)


def data_loader(x_all, batch_size, model_select, mode):
    x_input_ids = torch.tensor(x_all['input_ids'], dtype=torch.long)
    x_atten_masks = torch.tensor(x_all['attention_mask'], dtype=torch.long)
    y = torch.tensor(x_all['gt_label'], dtype=torch.long)
    if model_select == 'Bert':
        x_seg_ids = torch.tensor(x_all['token_type_ids'], dtype=torch.long)
        tensor_loader = TensorDataset(x_input_ids, x_atten_masks, x_seg_ids, y)
    else:
        tensor_loader = TensorDataset(x_input_ids, x_atten_masks, y)

    if mode == 'train':
        data_loader = DataLoader(tensor_loader, shuffle=True, batch_size=batch_size)
    else:
        data_loader = DataLoader(tensor_loader, shuffle=False, batch_size=batch_size)

    return data_loader, y
