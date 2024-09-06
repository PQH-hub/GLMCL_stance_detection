import torch
import torch.nn as nn
from transformers import BertModel, BartConfig, BartForSequenceClassification
from transformers.models.bart.modeling_bart import BartEncoder, BartPretrainedModel
from torch.nn import functional as F


# BERT
class bert_classifier(nn.Module):

    def __init__(self, num_labels, model_select, gen, dropout, dropoutrest):
        super(bert_classifier, self).__init__()

        self.dropout = nn.Dropout(dropout) if gen == 0 else nn.Dropout(dropoutrest)
        self.relu = nn.ReLU()

        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.bert.pooler = None
        self.linear = nn.Linear(self.bert.config.hidden_size * 2, self.bert.config.hidden_size)
        self.out = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, **kwargs):
        x_input_ids, x_atten_masks, x_seg_ids = kwargs['input_ids'], kwargs['attention_mask'], kwargs['token_type_ids']
        last_hidden = self.bert(input_ids=x_input_ids, attention_mask=x_atten_masks, token_type_ids=x_seg_ids)

        x_atten_masks[:, 0] = 0  # [CLS] --> 0
        idx = torch.arange(0, last_hidden[0].shape[1], 1).to('cuda')
        x_seg_ind = x_seg_ids * idx
        x_att_ind = (x_atten_masks - x_seg_ids) * idx
        indices_seg = torch.argmax(x_seg_ind, 1, keepdim=True)
        indices_att = torch.argmax(x_att_ind, 1, keepdim=True)
        for seg, seg_id, att, att_id in zip(x_seg_ids, indices_seg, x_atten_masks, indices_att):
            seg[seg_id] = 0  # 2nd [SEP] --> 0
            att[att_id:] = 0  # 1st [SEP] --> 0

        txt_l = x_atten_masks.sum(1).to('cuda')
        topic_l = x_seg_ids.sum(1).to('cuda')
        txt_vec = x_atten_masks.type(torch.FloatTensor).to('cuda')
        topic_vec = x_seg_ids.type(torch.FloatTensor).to('cuda')
        txt_mean = torch.einsum('blh,bl->bh', last_hidden[0], txt_vec) / txt_l.unsqueeze(1)
        topic_mean = torch.einsum('blh,bl->bh', last_hidden[0], topic_vec) / topic_l.unsqueeze(1)

        cat = torch.cat((txt_mean, topic_mean), dim=1)
        query = self.dropout(cat)
        linear = self.relu(self.linear(query))
        out = self.out(linear)

        return out


# BART
class Encoder(BartPretrainedModel):

    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)
        self.encoder = BartEncoder(config, self.shared)

    def forward(self, input_ids, attention_mask=None, output_attentions=False, output_hidden_states=False,
                return_dict=False):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return encoder_outputs


class bart_classifier(nn.Module):

    def __init__(self, num_labels, model_select, gen, dropout, dropoutrest):
        super(bart_classifier, self).__init__()

        self.dropout = nn.Dropout(dropout) if gen == 0 else nn.Dropout(dropoutrest)
        self.relu = nn.ReLU()

        self.config = BartConfig.from_pretrained(
            'C:\\Users\\20171\\Desktop\\Models\\bart-base')
        self.bart = Encoder.from_pretrained("C:\\Users\\20171\\Desktop\\Models\\bart-base")
        self.bart.pooler = None
        # 定义了两个线性层，linear用于将BART的隐藏层输出映射到更小的特征空间，out用于将特征空间映射到标签空间
        self.linear = nn.Linear(self.bart.config.hidden_size * 2, self.bart.config.hidden_size)
        self.out = nn.Linear(self.bart.config.hidden_size, num_labels)

    def forward(self, **kwargs):
        x_input_ids, x_atten_masks = kwargs['input_ids'], kwargs['attention_mask']
        cl_input_ids = torch.cat([x_input_ids, x_input_ids], dim=0)  # shape [batch_size*2,max_len]
        cl_atten_masks = torch.cat([x_atten_masks, x_atten_masks], dim=0)
        last_hidden = self.bart(input_ids=cl_input_ids, attention_mask=cl_atten_masks)

        # 调整了注意力掩码中与特定结束标记对应的位置，以控制模型在处理输入序列时的注意力分布
        eos_token_ind = cl_input_ids.eq(
            self.config.eos_token_id).nonzero()  # tensor([[0,4],[0,5],[0,11],[1,2],[1,3],[1,6]...])

        assert len(eos_token_ind) == 3 * len(cl_input_ids)
        b_eos = [eos_token_ind[i][1] for i in range(len(eos_token_ind)) if i % 3 == 0]
        e_eos = [eos_token_ind[i][1] for i in range(len(eos_token_ind)) if (i + 1) % 3 == 0]
        cl_atten_clone = cl_atten_masks.clone().detach()
        for begin, end, att, att2 in zip(b_eos, e_eos, cl_atten_masks, cl_atten_clone):
            att[begin:], att2[:begin + 2] = 0, 0  # att all </s> --> 0; att2 1st and 2nd </s> --> 0
            att[0], att2[end] = 0, 0  # <s> --> 0; 3rd </s> --> 0

        txt_l = cl_atten_masks.sum(1).to('cuda')
        topic_l = cl_atten_clone.sum(1).to('cuda')
        txt_vec = cl_atten_masks.type(torch.FloatTensor).to('cuda')
        topic_vec = cl_atten_clone.type(torch.FloatTensor).to('cuda')
        batch_size = int(last_hidden[0].shape[0] / 2)

        txt_mean = torch.einsum('blh,bl->bh', last_hidden[0], txt_vec) / txt_l.unsqueeze(1)
        topic_mean = torch.einsum('blh,bl->bh', last_hidden[0], topic_vec) / topic_l.unsqueeze(1)

        cat = torch.cat((txt_mean, topic_mean), dim=1)
        query = self.dropout(cat)
        linear = self.linear(query)
        linear1 = linear[:batch_size, ::]
        linear2 = linear[batch_size:linear.shape[0], ::]

        linear3 = torch.add(linear1, linear2)
        linear3 = torch.div(linear3, 2)

        features = linear.unsqueeze(1)  # batch_size*2,1,64
        features = F.normalize(features, dim=2)
        features = F.relu(features)

        out = self.out(self.relu(linear3))

        return out, features
