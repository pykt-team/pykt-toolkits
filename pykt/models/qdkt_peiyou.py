import os
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .peiyou_emb import QueBaseModelPeiyou, QuestionEncoder
from pykt.utils import debug_print

class QDKTNetPeiyou(nn.Module):
    def __init__(self, num_q,num_c,emb_size, dropout=0.1, emb_type='qaid', emb_path=dict(), pretrain_dim=768,device='cpu',mlp_layer_num=1,other_config={}):
        super().__init__()
        self.model_name = "qdkt_peiyou"
        self.num_q = num_q
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.device = device
        self.emb_type = emb_type
        
        self.que_emb = QuestionEncoder(num_q, emb_type, emb_size, dropout, emb_path, pretrain_dim)
        # self.que_emb = nn.Embedding(self.num_q, self.emb_size)

        self.interaction_emb = nn.Embedding(self.num_q * 2, self.emb_size)
        self.lstm_layer = nn.LSTM(self.emb_size, self.hidden_size, batch_first=True)
        self.dropout_layer = nn.Dropout(dropout)
        self.out_layer = nn.Linear(self.hidden_size, self.num_q)


    def forward(self, q, qtypes, c ,r,data=None):
        qemb = self.que_emb(q[:,:-1], qtypes[:,:-1])
        x = (q + self.num_q * r)[:,:-1]
        xemb = self.interaction_emb(x)
        xemb = xemb + qemb
        h, _ = self.lstm_layer(xemb)
        h = self.dropout_layer(h)
        y = self.out_layer(h)
        y = torch.sigmoid(y)
        y = (y * F.one_hot(data['qshft'].long(), self.num_q)).sum(-1)
        outputs = {"y":y}
        return outputs

class QDKTPeiyou(QueBaseModelPeiyou):
    def __init__(self, num_q,num_c, emb_size, dropout=0.1, emb_type='qaid', emb_path={}, pretrain_dim=768,device='cpu',seed=0,mlp_layer_num=1,other_config={},**kwargs):
        model_name = "qdkt_peiyou"
       
        debug_print(f"emb_type is {emb_type}",fuc_name="QDKTPeiyou")

        super().__init__(model_name=model_name,emb_type=emb_type,emb_path=emb_path,pretrain_dim=pretrain_dim)
        self.model = QDKTNetPeiyou(num_q=num_q,num_c=num_c,emb_size=emb_size,dropout=dropout,emb_type=emb_type,
                               emb_path=emb_path,pretrain_dim=pretrain_dim,device=device,mlp_layer_num=mlp_layer_num,other_config=other_config)
       
        self.model = self.model.to(device)
        self.emb_type = self.model.emb_type
        self.loss_func = self._get_loss_func("binary_crossentropy")
       
    def train_one_step(self,data,process=True,return_all=False):
        outputs,data_new = self.predict_one_step(data,return_details=True,process=process)
        loss = self.get_loss(outputs['y'],data_new['rshft'],data_new['sm'])
        return outputs['y'],loss#y_question没用

    def predict_one_step(self,data,return_details=False,process=True,return_raw=False):
        data_new = self.batch_to_device(data,process=process)
        outputs = self.model(data_new['cq'].long(),data_new['cqtypes'].long(),data_new['cc'],data_new['cr'].long(),data=data_new)
        if return_details:
            return outputs,data_new
        else:
            return outputs['y']