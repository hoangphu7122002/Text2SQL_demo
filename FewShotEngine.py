from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
import pandas as pd
import torch
import transformers
import os
import argparse
from functools import partial
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, Trainer, TrainingArguments, BitsAndBytesConfig, \
    DataCollatorForLanguageModeling, Trainer, TrainingArguments, BloomForCausalLM, BloomTokenizerFast, pipeline, \
    EarlyStoppingCallback
from huggingface_hub.hf_api import HfFolder
import random
from sentence_transformers import SentenceTransformer, util
import torch
from transformers import AutoModel, AutoTokenizer
import pickle

dict_ratio = {
    1 : [5],
    2 : [3,2],
    3 : [2,2,1]
}
LEVEL = 'syll'

def predict_top_k(y_proba, k, batch = False):
    if batch == True:
     top_k_predictions = []
    else:
     y_proba = [y_proba]
    for proba in y_proba:
    # Sort probabilities in descending order and get top k elements (classes with highest probabilities)
        proba = [(i,ele) for i,ele in enumerate(proba)]
        proba = [ele for ele in proba if ele[1] != 0]
        top_k_indices =  sorted(proba, key=lambda example: example[1], reverse=True)
        top_k_indices = top_k_indices[:min(len(top_k_indices),k)]
        top_k_indices = [ele[0] for ele in top_k_indices]
        if batch == True:
           top_k_predictions.append(top_k_indices)
    
    if batch == True:
      return top_k_predictions
    return top_k_indices

class inferFewShot:
    def __init__(self,level='syll'):
        self.level = level
        with open(f'knn_model_{level}.pkl', 'rb') as f:
            self.knn_model = pickle.load(f)

        self.phobert = AutoModel.from_pretrained("vinai/phobert-base-v2",device_map ="auto")
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
        self.dict_level = load_dataset(f"hoangphu7122002ai/autoFewshot_{level}", token='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')['train'].to_dict()

        print("===================CONSTRUCT DONE===================")
        
    def embedding_query(self,query):
        input_ids = torch.tensor([self.tokenizer.encode(query,truncation=True, max_length=254)])
        
        with torch.no_grad():
          features = self.phobert(input_ids.to('cuda:0'))
        
        ele = features[0][:, 0, :].to('cpu').numpy().squeeze(0)
        return ele
    
    def get_cluster(self,query,top_k=3, embedding_class=None):
        query_emb = self.embedding_query(query)
        predict_class = self.knn_model.predict_proba([query_emb])
    
        class_k = predict_top_k(predict_class,top_k,True)[0]
        dict_get = dict_ratio[len(class_k)]
        list_shot_all = []
        for len_sample,class_ele in zip(dict_get,class_k):
            idx_sample = random.sample(range(len(self.dict_level[f'{class_ele}_emb'])),500)
            test_matrix = []
            test_idx = {}
            test_shot = {}
            for j,k in enumerate(idx_sample):
                test_matrix.append(self.dict_level[f'{class_ele}_emb'][k])
                test_idx[j] = self.dict_level[f'{class_ele}_idx'][k]
                test_shot[j] = self.dict_level[f'{class_ele}_shot'][k]
            score_rerank = util.pytorch_cos_sim(query_emb,test_matrix)[0]
            top_k_max_indices = sorted(range(len(score_rerank)), key=lambda idx: score_rerank[idx], reverse=True)[:len_sample]
            list_idx = [test_shot[ele] for ele in top_k_max_indices]
            for sample in list_idx:
                list_shot_all.append(sample)
    
        return list_shot_all
