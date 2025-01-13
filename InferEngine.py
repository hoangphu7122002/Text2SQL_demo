import pandas as pd
import torch
import argparse
from functools import partial
import os
from huggingface_hub.hf_api import HfFolder
from vllm.lora.request import LoRARequest
from vllm import LLM, SamplingParams
from huggingface_hub import snapshot_download
import re
import random
from FewShotEngine import inferFewShot

HfFolder.save_token('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')

class SingletonMeta(type):
    """
    The Singleton class can be implemented in different ways in Python. Some
    possible methods include: base class, decorator, metaclass. We will use the
    metaclass because it is best suited for this purpose.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

class MyInstanceModel(metaclass=SingletonMeta):
    def __init__(self,model_name="codellama/CodeLlama-7b-Instruct-hf"):
        self.model = LLM(model=model_name,enable_lora=True,max_lora_rank=64,gpu_memory_utilization=0.3)
        self.engineShot = inferFewShot()
    
    def get_model(self):
        return self.model

    def get_engineShot(self):
        return self.engineShot

def extract_sql_queries(text):
    # Tìm tất cả các chuỗi nằm giữa các cặp dấu ``` trong chuỗi
    sql_queries = re.findall(r"```sql(.*?)```", text, re.DOTALL)
    return sql_queries

class InferEngine: 
    def __init__(self):
        self.model = self.load_model()
        self.sampling_params = self.set_sampling_param()
        self.lora_mode = {
            "express" : snapshot_download(repo_id='TeeA/ExpressSQL_codeLlama_text2sql_syll_r64_0'),
            "sql" : snapshot_download(repo_id='TeeA/codeLlama_text2sql_syll_r64_v2'),
            "CoT" : snapshot_download(repo_id='TeeA/CoT_codeLlama_text2sql_syll_r64_v0'),
            "FewShot" : snapshot_download(repo_id='TeeA/FewShot_codeLlama_text2sql_syll_r64_v0'),
        }
        self.dict_ref_lora = {
            "express" : (1,"express_adapter"),
            "sql" : (2,"sql_adapter"),
            "CoT" : (3,"cot_adapter"),
            "FewShot" : (4,"fewshot_adpater")
        }
        self.engineShot = MyInstanceModel().get_engineShot()
    
    def set_sampling_param(self,sampling=None):
        if sampling == None:
            return SamplingParams(
                        temperature=0,
                        top_k = 1,
                        max_tokens=1000,
                        stop=["</s>"]
                    )
        return sampling
    
    def load_model(self,model_name="codellama/CodeLlama-7b-Instruct-hf"):
        return MyInstanceModel(model_name).get_model()

    def get_prompt(self,input,mode,batched=False):
        if batched == False:
            schema = input['schema']
            question = input['question']
    
            if mode == 'sql':
                return f'''[INST] Sinh ra câu sql từ câu hỏi tương ứng với schema được cung cấp [/INST] ###schema: {schema}, ###câu hỏi: {question}, ###câu sql:'''
            elif mode == 'express':
                return f'''[INST] Diễn giải kết quả query sau thành câu trả lời tự nhiên dựa vào câu hỏi [/INST] ###query_result: {schema}, ###câu hỏi: {question}, ###câu trả lời tự nhiên được sinh ra:'''
            elif mode == 'CoT':
                return f'''[INST] Trình bày quá trình suy luận từ câu hỏi kết hợp schema để đưa ra câu truy vấn sql phù hợp [/INST] ###schema: {schema}, ###câu hỏi: {question}, ###CoT:'''
            else:
                shotList = self.engineShot.get_cluster(question)
                shotValue = random.sample(shotList,3)
                shotValue.append(f"""[INST] Sinh ra câu sql từ câu hỏi tương ứng với schema được cung cấp [/INST] ###schema: {schema}, ###câu hỏi: {question}, ###câu sql:""")
                return "\n".join(shotValue)
        else:
            pass
            
    def inference(self, input, mode):
        prompt = self.get_prompt(input,mode)
        name_ref_lora = self.dict_ref_lora[mode]
        lora_path = self.lora_mode[mode]

        
        result = self.model.generate(
            [prompt],
            self.sampling_params,
            lora_request=LoRARequest(name_ref_lora[1], name_ref_lora[0], lora_path)
        )
        if mode != 'CoT':
            return {"sql" : result[0].outputs[0].text}

        text = result[0].outputs[0].text
        query = extract_sql_queries(text)

        return {
            "CoT" : text,
            "sql" : query[0]
        }