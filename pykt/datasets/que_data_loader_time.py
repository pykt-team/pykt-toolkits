#!/usr/bin/env python
# coding=utf-8

import os, sys
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch import FloatTensor, LongTensor
import numpy as np

class KTQueDataset4PT(Dataset):
    """Dataset for KT
        can use to init dataset for: (for models except dkt_forget)
            train data, valid data
            common test data(concept level evaluation), real educational scenario test data(question level evaluation).

    Args:
        file_path (str): train_valid/test file path
        input_type (list[str]): the input type of the dataset, values are in ["questions", "concepts"]
        folds (set(int)): the folds used to generate dataset, -1 for test data
        qtest (bool, optional): is question evaluation or not. Defaults to False.
    """
    def __init__(self, file_path, input_type, folds,concept_num,max_concepts, qtest=False):
        super(KTQueDataset4PT, self).__init__()
        sequence_path = file_path
        self.input_type = input_type
        self.concept_num = concept_num
        self.max_concepts = max_concepts
        if "questions" not in input_type or "concepts" not in input_type:
            raise("The input types must contain both questions and concepts")

        folds = sorted(list(folds))
        folds_str = "_" + "_".join([str(_) for _ in folds])

        processed_data = file_path + folds_str + "_qlevel_pt.pkl"

        if not os.path.exists(processed_data):
            print(f"Start preprocessing {file_path} fold: {folds_str}...")

            self.dori, self.dgaps, self.max_sgap = self.__load_data__(sequence_path, folds)
            save_data = [self.dori, self.dgaps, self.max_sgap]
            pd.to_pickle(save_data, processed_data)
        else:
            print(f"Read data from processed file: {processed_data}")
            self.dori, self.dgaps, self.max_sgap = pd.read_pickle(processed_data)
        print(f"file path: {file_path}, qlen: {len(self.dori['qseqs'])}, clen: {len(self.dori['cseqs'])}, rlen: {len(self.dori['rseqs'])}")

    def __len__(self):
        """return the dataset length

        Returns:
            int: the length of the dataset
        """
        return len(self.dori["rseqs"])

    def __getitem__(self, index):
        """
        Args:
            index (int): the index of the data want to get

        Returns:
            (tuple): tuple containing:
            
            - **q_seqs (torch.tensor)**: question id sequence of the 0~seqlen-2 interactions
            - **c_seqs (torch.tensor)**: knowledge concept id sequence of the 0~seqlen-2 interactions
            - **r_seqs (torch.tensor)**: response id sequence of the 0~seqlen-2 interactions
            - **qshft_seqs (torch.tensor)**: question id sequence of the 1~seqlen-1 interactions
            - **cshft_seqs (torch.tensor)**: knowledge concept id sequence of the 1~seqlen-1 interactions
            - **rshft_seqs (torch.tensor)**: response id sequence of the 1~seqlen-1 interactions
            - **mask_seqs (torch.tensor)**: masked value sequence, shape is seqlen-1
            - **select_masks (torch.tensor)**: is select to calculate the performance or not, 0 is not selected, 1 is selected, only available for 1~seqlen-1, shape is seqlen-1
            - **dcur (dict)**: used only self.qtest is True, for question level evaluation
        """
        dcur = dict()
        mseqs = self.dori["masks"][index]
        for key in self.dori:
            if key in ["masks", "smasks"]:
                continue
            if len(self.dori[key]) == 0:
                dcur[key] = self.dori[key]
                dcur["shft_"+key] = self.dori[key]
                continue
            # print(f"key: {key}, len: {len(self.dori[key])}")
            if key=='cseqs':
                seqs = self.dori[key][index][:-1,:]
                shft_seqs = self.dori[key][index][1:,:]
            else:
                seqs = self.dori[key][index][:-1] * mseqs
                shft_seqs = self.dori[key][index][1:] * mseqs
            dcur[key] = seqs
            dcur["shft_"+key] = shft_seqs
        dcur["masks"] = mseqs
        dcur["smasks"] = self.dori["smasks"][index]
        # print("tseqs", dcur["tseqs"])
        dcurgaps = dict()
        for key in self.dgaps:
            seqs = self.dgaps[key][index][:-1] * mseqs
            shft_seqs = self.dgaps[key][index][1:] * mseqs
            dcurgaps[key] = seqs
            dcurgaps["shft_"+key] = shft_seqs
            
        return dcur, dcurgaps

    def get_skill_multi_hot(self, this_skills):
        skill_emb = [0] * self.concept_num
        for s in this_skills:
            skill_emb[s] = 1
        return skill_emb

    def __load_data__(self, sequence_path, folds, pad_val=-1):
        """
        Args:
            sequence_path (str): file path of the sequences
            folds (list[int]): 
            pad_val (int, optional): pad value. Defaults to -1.

        Returns: 
            (tuple): tuple containing

            - **q_seqs (torch.tensor)**: question id sequence of the 0~seqlen-1 interactions
            - **c_seqs (torch.tensor)**: knowledge concept id sequence of the 0~seqlen-1 interactions
            - **r_seqs (torch.tensor)**: response id sequence of the 0~seqlen-1 interactions
            - **mask_seqs (torch.tensor)**: masked value sequence, shape is seqlen-1
            - **select_masks (torch.tensor)**: is select to calculate the performance or not, 0 is not selected, 1 is selected, only available for 1~seqlen-1, shape is seqlen-1
            - **dqtest (dict)**: not null only self.qtest is True, for question level evaluation
        """
        dori = {"qseqs": [], "cseqs": [], "rseqs": [], "tseqs": [], "utseqs": [], "smasks": []}
        dgaps = {"sgaps": [], "pretlabel":[], "citlabel":[]}
        max_sgap = 0

        df = pd.read_csv(sequence_path)
        df = df[df["fold"].isin(folds)].copy()#[0:1000]
        interaction_num = 0
        for i, row in df.iterrows():
            #use kc_id or question_id as input
            if "concepts" in self.input_type:
                row_skills = []
                raw_skills = row["concepts"].split(",")
                for concept in raw_skills:
                    if concept == "-1":
                        skills = [-1] * self.max_concepts
                    else:
                        skills = [int(_) for _ in concept.split("_")]
                        skills = skills +[-1]*(self.max_concepts-len(skills))
                    row_skills.append(skills)
                dori["cseqs"].append(row_skills)
            if "questions" in self.input_type:
                try:
                    dori["qseqs"].append([int(_) for _ in row["questions"].split(",")])
                except:
                    que_seq = row["questions"]
                    print(f"i:{i}, questions:{que_seq}")
            if "timestamps" in row:
                dori["tseqs"].append([int(_) for _ in row["timestamps"].split(",")])
            if "usetimes" in row:
                dori["utseqs"].append([int(_) for _ in row["usetimes"].split(",")])
                
            dori["rseqs"].append([int(_) for _ in row["responses"].split(",")])
            dori["smasks"].append([int(_) for _ in row["selectmasks"].split(",")])

            # add temporal info
            if sequence_path.find("assist2009") == -1 and sequence_path.find("assist2015") == -1:
                sgap, pret_label, cit_label = self.calC(row)
            else:
                sgap, pret_label, cit_label = self.calC_INDEX(row)
                
            dgaps["sgaps"].append(sgap)
            dgaps["pretlabel"].append(pret_label)
            dgaps["citlabel"].append(cit_label)
            
            max_sgap = max(sgap) if max(sgap) > max_sgap else max_sgap

            interaction_num += dori["smasks"][-1].count(1)


        for key in dori:
            if key not in ["rseqs"]:#in ["smasks", "tseqs"]:
                dori[key] = LongTensor(dori[key])
            else:
                dori[key] = FloatTensor(dori[key])

        mask_seqs = (dori["rseqs"][:,:-1] != pad_val) * (dori["rseqs"][:,1:] != pad_val)
        dori["masks"] = mask_seqs
        
        dori["smasks"] = (dori["smasks"][:, 1:] != pad_val)
        
        for key in dgaps:
            if key not in ["pretlabel", "citlabel"]:
                # print(f"key:{key},  {dgaps[key]}")
                dgaps[key] = LongTensor(dgaps[key])
            else:
                dgaps[key] = FloatTensor(dgaps[key])
                
        print(f"interaction_num: {interaction_num}")
        # print("load data tseqs: ", dori["tseqs"])
        return dori, dgaps, max_sgap
    
    def log2(self, t):
        import math
        return round(math.log(t+1, 2))

    def calC(self, row):
        sequence_gap, pret_label, cit_label = [], [], []
        uid = row["uid"]
        # default: concepts
        skills = row["concepts"].split(",") if "concepts" in self.input_type else row["questions"].split(",")
        timestamps = row["timestamps"].split(",")
        dpreskill, dlastskill, dcount = dict(), dict(), dict()
        pret, double_pret = None, None
        cnt = 0
        for idx,(s, t) in enumerate(zip(skills, timestamps)):
            s, t = int(s), int(t)
            if s not in dlastskill or s == -1:
                curCIt = 0
                dlastskill[s] = -1
            else:
                if dpreskill[s] == -1:
                    curCIt = 0.5
                else:
                    precit = int((t - dlastskill[s]))/1000
                    double_precit = int((t - dpreskill[s]))/1000
                    curCIt = round(1 - ((precit + 0.01)/(double_precit + 0.01)),2)
            dpreskill[s] = dlastskill[s]
            dlastskill[s] = t

            cit_label.append(curCIt)

            if pret == None or t == -1:
                if t == -1:
                    cnt += 1
                curLastGap = 0
                curLastIt = 0
                curPreT = 0
                if idx == 0:
                    curLableT = 0
                    double_pret = t
                else:
                    if cnt == 2:
                        t_label[-1] = 1
                    else:
                        curLableT = 0
            else:
                curLastGap = self.log2((t - pret) / 1000 / 60) + 1
                # curLastIt = min(int((t - pret) / 1000 / 60) + 1,43200)
                curLastIt = int((t - pret)) / 1000
                curPreIt = int((pret - double_pret))/1000
                curPostIt = int((t - double_pret))/1000
                curLableT = round((curPreIt+0.01)/(curPostIt+0.01),2)
                curPreT = round(1 - ((curLastIt + 0.01)/(curPostIt + 0.01)),2)
                double_pret = pret
            pret = t
            sequence_gap.append(curLastGap)
            pret_label.append(curPreT)
            
        pret_label = [0, 0.5]+pret_label[2:]
    
        return sequence_gap, pret_label, cit_label
            

    def calC_INDEX(self, row):
        sequence_gap, pret_label, cit_label = [], [], []
        uid = row["uid"]
        # default: concepts
        skills = row["concepts"].split(",") if "concepts" in self.input_type else row["questions"].split(",")
        timestamps = [i for i in range(len(skills))]
        dpreskill, dlastskill, dcount = dict(), dict(), dict()
        pret, double_pret = None, None
        cnt = 0

        for idx,(s, t) in enumerate(zip(skills, timestamps)):
            s, t = int(s), int(t)
            if s not in dlastskill or s == -1:
                curCIt = 0
                dlastskill[s] = -1
            else:
                if dpreskill[s] == -1:
                    # print(f"s:{s}")
                    curCIt = 0.5
                else:
                    # print(f"s:{s}")
                    precit = int((t - dlastskill[s]))
                    double_precit = int((t - dpreskill[s]))
                    curCIt = round(1 - ((precit + 0.01)/(double_precit + 0.01)),2)
            dpreskill[s] = dlastskill[s]
            dlastskill[s] = t

            cit_label.append(curCIt)

            if pret == None or t == -1:
                if t == -1:
                    cnt += 1
                curLastGap = 0
                curLastIt = 0
                curPreT = 0
                if idx == 0:
                    curLableT = 0
                    double_pret = t
                else:
                    if cnt == 2:
                        t_label[-1] = 1
                    else:
                        curLableT = 0

            else:
                curLastGap = (t - pret) 
                curLastIt = int((t - pret)) 
                curPreIt = int((pret - double_pret))
                curPostIt = int((t - double_pret))
                curLableT = round((curPreIt+0.01)/(curPostIt+0.01),2)
                curPreT = round(1 - ((curLastIt + 0.01)/(curPostIt + 0.01)),2)
                double_pret = pret
            pret = t
            sequence_gap.append(curLastGap)
            pret_label.append(curPreT)

        pret_label = [0, 0.5]+pret_label[2:]
    
        return sequence_gap, pret_label, cit_label
