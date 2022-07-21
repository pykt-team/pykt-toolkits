import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
import math
import torch.nn.functional as F
from enum import IntEnum
import numpy as np
from torch.nn import Module, Embedding, LSTM, Linear, Dropout, LayerNorm, TransformerEncoder, TransformerEncoderLayer, MultiLabelMarginLoss, MultiLabelSoftMarginLoss, CrossEntropyLoss
from .utils import transformer_FFN, ut_mask, pos_encode, get_clones
from .sakt import Blocks

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2

class CAKT(nn.Module):
    def __init__(self, n_question, n_pid, seq_len, d_model, n_blocks, dropout, loss1=0.5, loss2=0.5, loss3=0.5,
            d_ff=256, 
            kq_same=1, final_fc_dim=512, num_attn_heads=8, separate_qa=False, l2=1e-5, emb_type="qid", emb_path="", pretrain_dim=768):
        super().__init__()
        """
        Input:
            d_model: dimension of attention block
            final_fc_dim: dimension of final fully connected net before prediction
            num_attn_heads: number of heads in multi-headed attention
            d_ff : dimension for fully conntected net inside the basic block
            kq_same: if key query same, kq_same=1, else = 0
        """
        self.model_name = "cakt"
        self.n_question = n_question
        self.dropout = dropout
        self.kq_same = kq_same
        self.n_pid = n_pid
        self.rashl2 = l2
        self.model_type = self.model_name
        self.separate_qa = separate_qa
        self.emb_type = emb_type
        embed_l = d_model
        if self.n_pid > 0:
            self.difficult_param = nn.Embedding(self.n_pid+1, 1) # 题目难度
            self.q_embed_diff = nn.Embedding(self.n_question+1, embed_l) # question emb, 总结了包含当前question（concept）的problems（questions）的变化
            self.qa_embed_diff = nn.Embedding(2 * self.n_question + 1, embed_l) # interaction emb, 同上
        
        if emb_type.startswith("qid"):
            # n_question+1 ,d_model
            # self.q_embed = nn.Embedding(self.n_question, embed_l)
            self.q_embed = nn.Parameter(torch.randn(self.n_question, embed_l).to(device), requires_grad=True)
            if self.separate_qa: 
                self.qa_embed = nn.Embedding(2*self.n_question+1, embed_l) # interaction emb
            else: # false default
                self.qa_embed = nn.Embedding(2, embed_l)


        # Architecture Object. It contains stack of attention block
        self.model = Architecture(n_question=n_question, n_blocks=n_blocks, n_heads=num_attn_heads, dropout=dropout,
                                    d_model=d_model, d_feature=d_model / num_attn_heads, d_ff=d_ff,  kq_same=self.kq_same, model_type=self.model_type)

        self.out = nn.Sequential(
            nn.Linear(d_model + embed_l,
                      final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, 256), nn.ReLU(
            ), nn.Dropout(self.dropout),
            nn.Linear(256, 1)
        )

        if emb_type.endswith("sharepredcurc"):
            self.l1 = loss1
            self.l2 = loss2
            self.question_emb = nn.Embedding(self.n_pid, embed_l) # 1.2
            self.qclasifier = nn.Sequential(
                nn.Linear(d_model + embed_l + embed_l,
                        final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
                nn.Linear(final_fc_dim, 256), nn.ReLU(
                ), nn.Dropout(self.dropout),
                nn.Linear(256, self.n_question)
            )
            self.out = nn.Sequential(
                nn.Linear(d_model + embed_l + embed_l,
                        final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
                nn.Linear(final_fc_dim, 256), nn.ReLU(
                ), nn.Dropout(self.dropout),
                nn.Linear(256, 1)
            )

        if emb_type.endswith("predcurc"):
            self.l1 = loss1
            self.l2 = loss2
            self.question_emb = nn.Embedding(self.n_pid, embed_l) # 1.2
            self.qlstm = nn.LSTM(embed_l, embed_l, batch_first=True)
            
            self.qdrop = nn.Dropout(dropout)
            self.qclasifier = nn.Linear(embed_l, self.n_question)

        if emb_type.endswith("mergetwo"):
            self.l1, self.l2, self.l3 = loss1, loss2, loss3
            self.nhead = num_attn_heads
            encoder_layer1 = TransformerEncoderLayer(d_model, nhead=self.nhead)
            encoder_norm1 = LayerNorm(d_model)
            self.embed_l = embed_l

            if self.n_pid > 0:
                self.question_emb = Embedding(self.n_pid, embed_l) # 1.2

            if self.emb_type.find("trans") != -1:
                self.qnet1 = TransformerEncoder(encoder_layer1, num_layers=2, norm=encoder_norm1)
            else:    
                self.qnet1 = LSTM(embed_l, d_model, batch_first=True)
            # self.qdrop1 = Dropout(dropout)
            self.qclasifier1 = Linear(d_model, self.n_question)
            self.closs1 = CrossEntropyLoss()

            # seq2seq
            self.position_emb = Embedding(seq_len, embed_l)
            encoder_layer2 = TransformerEncoderLayer(d_model, nhead=self.nhead)
            encoder_norm2 = LayerNorm(d_model)
            self.base_qnet21 = TransformerEncoder(encoder_layer2, num_layers=1, norm=encoder_norm2)
            self.base_qnet22 = TransformerEncoder(encoder_layer2, num_layers=n_blocks, norm=encoder_norm2)
            self.qnet2 = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout)
            )
            self.qclasifier2 = nn.Linear(d_model, self.n_question)
            self.closs2 = MultiLabelSoftMarginLoss()# MultiLabelMarginLoss()   

            # predict kt
            self.model = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // num_attn_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=num_attn_heads, kq_same=kq_same)
                for _ in range(n_blocks)
            ])
            # self.model = get_clones(Blocks(d_model, num_attn_heads, dropout), n_blocks)
            # self.n_blocks = n_blocks
            # encoder_layer3 = TransformerEncoderLayer(d_model, nhead=self.nhead)
            # encoder_norm3 = LayerNorm(d_model)
            # self.model = TransformerEncoder(encoder_layer3, num_layers=n_blocks, norm=encoder_norm3) # q&k: seq2seq' hidden; v: predcurc' hidden
        
        self.reset()

    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.n_pid+1 and self.n_pid > 0:
                torch.nn.init.constant_(p, 0.)

    def base_emb(self, q_data, target):
        q_embed_data = self.q_embed[q_data]  # BS, seqlen,  d_model# c_ct
        if self.separate_qa:
            qa_data = q_data + self.n_question * target
            qa_embed_data = self.qa_embed(qa_data)
        else:
            # BS, seqlen, d_model # c_ct+ g_rt =e_(ct,rt)
            qa_embed_data = self.qa_embed(target)+q_embed_data
        return q_embed_data, qa_embed_data

    def get_avg_skill_emb(self, c):
        # add zero for padding
        concept_emb_cat = torch.cat(
            [torch.zeros(1, self.embed_l).to(device), 
            self.q_embed], dim=0)
        # shift c
        related_concepts = (c+1).long()
        #[batch_size, seq_len, emb_dim]
        concept_emb_sum = concept_emb_cat[related_concepts, :].sum(
            axis=-2).to(device)

        #[batch_size, seq_len,1]
        concept_num = torch.where(related_concepts != 0, 1, 0).sum(
            axis=-1).unsqueeze(-1)
        concept_num = torch.where(concept_num == 0, 1, concept_num).to(device)
        concept_avg = (concept_emb_sum / concept_num)
        return concept_avg

    def get_attn_pad_mask(self, sm):
        batch_size, l = sm.size()
        pad_attn_mask = sm.data.eq(0).unsqueeze(1)
        pad_attn_mask = pad_attn_mask.expand(batch_size, l, l)
        return pad_attn_mask.repeat(self.nhead, 1, 1)

    def predcurc(self, repqemb, repcemb, cc, sm, emb_type, xemb, train):
        # predcurc
        y2 = 0
        if self.n_pid > 0:
            catemb = xemb + repqemb
        else:
            catemb = xemb
        if emb_type.find("cemb") != -1:
            catemb += repcemb
        if emb_type.find("trans") != -1:
            mask = ut_mask(seq_len = catemb.shape[1])
            qh = self.qnet1(catemb.transpose(0,1), mask).transpose(0,1)
        else:
            qh, _ = self.qnet1(catemb)
        if train:
            cpreds = self.qclasifier1(qh) # 之前版本没加sigmoid
            padsm = torch.ones(sm.shape[0], 1).to(device)
            sm = torch.cat([padsm, sm], dim=-1)
            flag = sm==1
            y2 = self.closs1(cpreds[flag], cc[flag])
        xemb1 = xemb + qh
        return y2, xemb1

    def predmultilabel(self, repcemb, repqemb, dcur, train):
        y3 = 0
        if train:
            oriqs, orics, orisms = dcur["oriqs"].long(), dcur["orics"].long(), dcur["orisms"].long()#self.generate_oriqcs(q, c, sm)
            oriqshft, oricshft = dcur["shft_oriqs"].long(), dcur["shft_orics"].long()
            oriqs, orics = torch.cat([oriqs[:,0:1], oriqshft], dim=-1), torch.cat([orics[:,0:1,:], oricshft], dim=1)
            concept_avg = self.get_avg_skill_emb(orics)
            qemb = self.question_emb(oriqs)
            oriposemb = self.position_emb(pos_encode(qemb.shape[1]))
            # print(f"concept_avg: {concept_avg.shape}, qemb: {qemb.shape}, posemb: {posemb.shape}")
            que_c_emb = concept_avg + qemb + oriposemb#torch.cat([concept_avg, qemb],dim=-1)

            # add mask
            # mask = self.get_attn_pad_mask(orisms)
            mask = ut_mask(seq_len = que_c_emb.shape[1])
            qh = self.qnet2(self.base_qnet22(self.base_qnet21(que_c_emb.transpose(0,1), mask).transpose(0,1)))
            cpreds = torch.sigmoid(self.qclasifier2(qh))
            padsm = torch.ones(orisms.shape[0], 1).to(device)
            orisms = torch.cat([padsm, orisms], dim=-1)
            flag = orisms==1
            masked = cpreds[flag]
            pad = torch.ones(cpreds.shape[0], cpreds.shape[1], self.n_question-10).to(device)
            pad = -1 * pad
            ytrues = torch.cat([orics, pad], dim=-1).long()[flag]
            y3 = self.closs2(masked, ytrues)
        posemb = self.position_emb(pos_encode(repcemb.shape[1]))
        qcemb = repcemb+repqemb+posemb#torch.cat([repcemb, repqemb], dim=-1)
        # qcmask = self.get_attn_pad_mask(sm)
        qcmask = ut_mask(seq_len = qcemb.shape[1])
        qcemb = self.qnet2(self.base_qnet22(self.base_qnet21(qcemb.transpose(0,1), qcmask).transpose(0,1)))
        xemb2 = qcemb + repcemb+repqemb
        return y3, xemb2

    def forward(self, dcur, qtest=False, train=False):#q_data, target, pid_data=None, qtest=False, train=False):
        q, c, r = dcur["qseqs"].long(), dcur["cseqs"].long(), dcur["rseqs"].long()
        qshft, cshft, rshft = dcur["shft_qseqs"].long(), dcur["shft_cseqs"].long(), dcur["shft_rseqs"].long()
        sm = dcur["smasks"]
        pid_data = torch.cat((q[:,0:1], qshft), dim=1)
        q_data = torch.cat((c[:,0:1], cshft), dim=1)
        target = torch.cat((r[:,0:1], rshft), dim=1)

        emb_type = self.emb_type
        # Batch First
        if emb_type.startswith("qid"):
            q_embed_data, qa_embed_data = self.base_emb(q_data, target)

        if self.n_pid > 0 and not emb_type.endswith("mergetwo"): # have problem id
            q_embed_diff_data = self.q_embed_diff(q_data)  # d_ct 总结了包含当前question（concept）的problems（questions）的变化
            pid_embed_data = self.difficult_param(pid_data)  # uq 当前problem的难度
            q_embed_data = q_embed_data + pid_embed_data * \
                q_embed_diff_data  # uq *d_ct + c_ct # question encoder

            qa_embed_diff_data = self.qa_embed_diff(
                target)  # f_(ct,rt) or #h_rt (qt, rt)差异向量
            if self.separate_qa:
                qa_embed_data = qa_embed_data + pid_embed_data * \
                    qa_embed_diff_data  # uq* f_(ct,rt) + e_(ct,rt)
            else:
                qa_embed_data = qa_embed_data + pid_embed_data * \
                    (qa_embed_diff_data+q_embed_diff_data)  # + uq *(h_rt+d_ct) # （q-response emb diff + question emb diff）
            c_reg_loss = (pid_embed_data ** 2.).sum() * self.rashl2 # rasch部分loss
        else:
            c_reg_loss = 0.


        ### 
        y2, y3 = 0, 0
        if emb_type == "qid":
            d_output = self.model(q_embed_data, qa_embed_data)

            concat_q = torch.cat([d_output, q_embed_data], dim=-1)
            output = self.out(concat_q).squeeze(-1)
            m = nn.Sigmoid()
            preds = m(output)
        elif emb_type.endswith("mergetwo"): #
            if self.n_pid > 0:
                repqemb = self.question_emb(pid_data)
            repcemb = self.q_embed[q_data] # 原来的q_embed_data
            if emb_type.find("predcurc") != -1:
                y2, xemb1 = self.predcurc(repqemb, repcemb, q_data, sm, emb_type, qa_embed_data, train)
            if emb_type.find("ml") != -1:
                y3, xemb2 = self.predmultilabel(repcemb, repqemb, dcur, train)
            q_embed_data, qa_embed_data = xemb2, xemb1

            # TODO!!!
            x, y = q_embed_data, qa_embed_data
            for block in self.model:
                if emb_type.find("decay") != -1:
                    x = block(mask=0, query=x, key=x, values=y, apply_pos=True, decay=True) # True: +FFN+残差+laynorm 非第一层与0~t-1的的q的attention, 对应图中Knowledge Retriever
                else:
                    x = block(mask=0, query=x, key=x, values=y, apply_pos=True, decay=False)
            # value = qa_embed_data
            # for i in range(self.n_blocks):
            #     value = self.model[i](q_embed_data, q_embed_data, value)
            # d_output = value
            # d_output = self.model(q_embed_data, qa_embed_data)
            concat_q = torch.cat([x, q_embed_data], dim=-1)
            output = self.out(concat_q).squeeze(-1)
            m = nn.Sigmoid()
            preds = m(output)
            
        elif emb_type.endswith("sharepredcurc"):
            d_output = self.model(q_embed_data, qa_embed_data)
            concat_q = torch.cat([d_output, q_embed_data], dim=-1)

            qemb = self.question_emb(pid_data)
            concat_qemb = torch.cat([concat_q, qemb], dim=-1)
            cout = self.qclasifier(concat_qemb).squeeze(-1)
            m = nn.Sigmoid()
            y2 = m(cout)

            output = self.out(concat_qemb).squeeze(-1)
            m = nn.Sigmoid()
            preds = m(output)
        elif emb_type.endswith("predcurc"): # predict current question' current concept
            # predict concept
            qemb = self.question_emb(pid_data)

            pad = torch.zeros(qa_embed_data.shape[0], 1, qa_embed_data.shape[2]).to(device)
            chistory = torch.cat((pad, qa_embed_data[:,0:-1,:]), dim=1)
            # chistory = qa_embed_data
            catemb = qemb + chistory
            if emb_type.find("cemb") != -1:
                cemb = q_embed_data
                catemb += cemb
            qh, _ = self.qlstm(catemb)
            y2 = self.qclasifier(qh)

            # predict response
            # pad = torch.zeros(xemb.shape[0], 1, xemb.shape[2]).to(device)
            # chistory = torch.cat((pad, xemb[:,0:-1,:]), dim=1)
            qa_embed_data = qa_embed_data + qh + cemb if emb_type.find("cemb") != -1 else qa_embed_data + qh
            # if emb_type.find("qemb") != -1:
            #     qa_embed_data += qemb
            d_output = self.model(q_embed_data, qa_embed_data)

            if emb_type.find("addfinal") != -1:
                d_output += cemb + qh
            concat_q = torch.cat([d_output, q_embed_data], dim=-1)
            output = self.out(concat_q).squeeze(-1)
            m = nn.Sigmoid()
            preds = m(output)
        if train:
            return preds, c_reg_loss, y2, y3
        else:
            if not qtest:
                return preds, c_reg_loss
            else:
                return preds, c_reg_loss, concat_q


class Architecture(nn.Module):
    def __init__(self, n_question,  n_blocks, d_model, d_feature,
                 d_ff, n_heads, dropout, kq_same, model_type):
        super().__init__()
        """
            n_block : number of stacked blocks in the attention
            d_model : dimension of attention input/output
            d_feature : dimension of input in each of the multi-head attention part.
            n_head : number of heads. n_heads*d_feature = d_model
        """
        self.d_model = d_model
        self.model_type = model_type

        if model_type in {'cakt'}:
            self.blocks_1 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
                for _ in range(n_blocks)
            ])
            self.blocks_2 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
                for _ in range(n_blocks*2)
            ])

    def forward(self, q_embed_data, qa_embed_data):
        # target shape  bs, seqlen
        seqlen, batch_size = q_embed_data.size(1), q_embed_data.size(0)

        qa_pos_embed = qa_embed_data
        q_pos_embed = q_embed_data

        y = qa_pos_embed
        seqlen, batch_size = y.size(1), y.size(0)
        x = q_pos_embed

        # encoder
        for block in self.blocks_1:  # encode qas, 对0～t-1时刻前的qa信息进行编码
            y = block(mask=1, query=y, key=y, values=y) # yt^
        flag_first = True
        for block in self.blocks_2:
            if flag_first:  # peek current question
                x = block(mask=1, query=x, key=x,
                          values=x, apply_pos=False) # False: 没有FFN, 第一层只有self attention, 对应于xt^
                flag_first = False
            else:  # dont peek current response
                x = block(mask=0, query=x, key=x, values=y, apply_pos=True) # True: +FFN+残差+laynorm 非第一层与0~t-1的的q的attention, 对应图中Knowledge Retriever
                # mask=0，不能看到当前的response, 在Knowledge Retrever的value全为0，因此，实现了第一题只有question信息，无qa信息的目的
                # print(x[0,0,:])
                flag_first = True
        return x

class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature,
                 d_ff, n_heads, dropout,  kq_same):
        super().__init__()
        """
            This is a Basic Block of Transformer paper. It containts one Multi-head attention object. Followed by layer norm and postion wise feedforward net and dropout layer.
        """
        kq_same = kq_same == 1
        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same)

        # Two layer norm layer and two droput layer
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask, query, key, values, apply_pos=True, decay=True):
        """
        Input:
            block : object of type BasicBlock(nn.Module). It contains masked_attn_head objects which is of type MultiHeadAttention(nn.Module).
            mask : 0 means, it can peek only past values. 1 means, block can peek only current and pas values
            query : Query. In transformer paper it is the input for both encoder and decoder
            key : Keys. In transformer paper it is the input for both encoder and decoder
            Values. In transformer paper it is the input for encoder and  encoded output for decoder (in masked attention part)

        Output:
            query: Input gets changed over the layer and returned.

        """

        seqlen, batch_size = query.size(1), query.size(0)
        nopeek_mask = np.triu(
            np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8')
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)
        if mask == 0:  # If 0, zero-padding is needed.
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=True, decay=decay) # 只能看到之前的信息，当前的信息也看不到，此时会把第一行score全置0，表示第一道题看不到历史的interaction信息，第一题attn之后，对应value全0
        else:
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=False, decay=decay)

        query = query + self.dropout1((query2)) # 残差1
        query = self.layer_norm1(query) # layer norm
        if apply_pos:
            query2 = self.linear2(self.dropout( # FFN
                self.activation(self.linear1(query))))
            query = query + self.dropout2((query2)) # 残差
            query = self.layer_norm2(query) # lay norm
        return query


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True):
        super().__init__()
        """
        It has projection layer for getting keys, queries and values. Followed by attention and a connected layer.
        """
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same

        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same is False:
            self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        torch.nn.init.xavier_uniform_(self.gammas)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if self.kq_same is False:
            xavier_uniform_(self.q_linear.weight)

        if self.proj_bias:
            constant_(self.k_linear.bias, 0.)
            constant_(self.v_linear.bias, 0.)
            if self.kq_same is False:
                constant_(self.q_linear.bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, q, k, v, mask, zero_pad, decay):

        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.kq_same is False:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        gammas = self.gammas
        scores = attention(q, k, v, self.d_k,
                           mask, self.dropout, zero_pad, gammas, decay)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous()\
            .view(bs, -1, self.d_model)

        output = self.out_proj(concat)

        return output


def attention(q, k, v, d_k, mask, dropout, zero_pad, gamma=None, decay=True):
    """
    This is called by Multi-head atention object to find the values.
    """
    # d_k: 每一个头的dim
    scores = torch.matmul(q, k.transpose(-2, -1)) / \
        math.sqrt(d_k)  # BS, 8, seqlen, seqlen
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    if decay:
        x1 = torch.arange(seqlen).expand(seqlen, -1).to(device)
        x2 = x1.transpose(0, 1).contiguous()

        with torch.no_grad():
            scores_ = scores.masked_fill(mask == 0, -1e32)
            scores_ = F.softmax(scores_, dim=-1)  # BS,8,seqlen,seqlen
            scores_ = scores_ * mask.float().to(device) # 结果和上一步一样
            distcum_scores = torch.cumsum(scores_, dim=-1)  # bs, 8, sl, sl
            disttotal_scores = torch.sum(
                scores_, dim=-1, keepdim=True)  # bs, 8, sl, 1 全1
            # print(f"distotal_scores: {disttotal_scores}")
            position_effect = torch.abs(
                x1-x2)[None, None, :, :].type(torch.FloatTensor).to(device)  # 1, 1, seqlen, seqlen 位置差值
            # bs, 8, sl, sl positive distance
            dist_scores = torch.clamp(
                (disttotal_scores-distcum_scores)*position_effect, min=0.) # score <0 时，设置为0
            dist_scores = dist_scores.sqrt().detach()
        m = nn.Softplus()
        gamma = -1. * m(gamma).unsqueeze(0)  # 1,8,1,1 一个头一个gamma参数， 对应论文里的theta
        # Now after do exp(gamma*distance) and then clamp to 1e-5 to 1e5
        total_effect = torch.clamp(torch.clamp(
            (dist_scores*gamma).exp(), min=1e-5), max=1e5) # 对应论文公式1中的新增部分
        scores = scores * total_effect

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen
    # print(f"before zero pad scores: {scores.shape}")
    # print(zero_pad)
    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2) # 第一行score置0
    # print(f"after zero pad scores: {scores}")
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    # import sys
    # sys.exit()
    return output


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=True)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)


class CosinePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)
