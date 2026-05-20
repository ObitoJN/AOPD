# -*- coding: utf-8 -*-
import json
import time
import requests
import math
from typing import List, Dict, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import random
from openai import OpenAI
# ==========================================
# 1. API 接口类定义
# ==========================================

import ipaddress

def validate_ip(ip_list):
    invalid_ips = []
    valid_ips = []

    for ip in ip_list:
        if not ip:
            continue
        try:
            ipaddress.ip_address(ip)
            valid_ips.append(ip)
        except ValueError:
            invalid_ips.append(ip)

    # 3. 返回结果
    if len(invalid_ips) == 0:
        return True, valid_ips
    else:
        return False, invalid_ips

class vllmAPIModelInterface:
    def __init__(self,
                 app_id: str = "your_api_key",
                 model_name: str = "",
                 max_tokens: int = 2000,
                 max_retries: int = 3,
                 timeout: int = None,
                 top_k: int = 32,
                 temperature: float = 1.0,
                 ip_pool: list = [],
                 **kwargs):
        self.app_id = app_id
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.timeout = timeout
        self.top_k = top_k
        self.temperature = temperature

        final_ip_pool = []
        ip_flag, final_ip_pool = validate_ip(ip_pool)

        if len(final_ip_pool) == 0:
            self.ip_pool = ['127.0.0.1']
            print(f"未识别到输入IP池, 采用默认IP池: {self.ip_pool}")
        elif len(final_ip_pool)!=0 and not ip_flag:
            raise ValueError(f"存在非法IP, 非法IP为: {final_ip_pool}")
        else:
            print(f"识别到有效输入IP池, 采用输入IP池: {final_ip_pool}")
            self.ip_pool = final_ip_pool

        self.config = kwargs
        self._initialize_client()

    def _initialize_client(self):
        print(f"API客户端初始化完成 - 模型: {self.model_name}")
        self.client_pool = []
        for item in self.ip_pool:
            client = OpenAI(
                base_url="http://{}:8000/v1".format(item),
                api_key='your_api_key',
                timeout=self.timeout  # None 表示无限超时
            )
            self.client_pool.append(client)

    def get_single_answer(self, prompt: str) -> Union[str, Dict[str, Any]]:
        for attempt in range(self.max_retries):
            try:
                result = self._call_api_single(prompt)
                return result
            except Exception as e:
                if attempt == self.max_retries - 1:
                    # 最后一次重试失败后，抛出异常而不是返回错误字典
                    raise RuntimeError(
                        f"Teacher API 调用失败，已重试 {self.max_retries} 次。"
                        f"错误信息: {str(e)}。"
                        f"请检查 API 服务是否正常运行。"
                    ) from e
                time.sleep(1)

    def get_batch_answers(self, prompts: List[str], max_workers: int = 100) -> List[Union[str, Dict[str, Any]]]:
        if not prompts:
            return []

        answers = [None] * len(prompts)
        total = len(prompts)
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(self.get_single_answer, prompt): i 
                for i, prompt in enumerate(prompts)
            }
            
            # completed_count = 0
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    answers[index] = future.result()
                except Exception as e:
                    # 批量调用中任何一个失败都应该抛出异常
                    raise RuntimeError(
                        f"Teacher API 批量调用失败。"
                        f"失败的样本索引: {index}，"
                        f"错误信息: {str(e)}。"
                        f"请检查 API 服务是否正常运行。"
                    ) from e

        return answers

    def _call_api_single(self, prompt: str) -> Union[str, Dict[str, Any]]:
        sample_client = random.choice(self.client_pool)

        response = sample_client.completions.create(
                model=self.model_name,
                prompt=prompt,
                logprobs=self.top_k,
                max_tokens=1,
                temperature=self.temperature,
                echo=True,
                timeout=self.timeout  # None 表示无限超时
            )
            
        choice = response.choices[0]

        try:
            if not choice:
                return {
                    "prompt": ["Error: No choices"],
                    "prompt_logprobs": [0.0],
                    "topk_indices": np.zeros((1, self.top_k), dtype=int),
                    "topk_logprobs": np.zeros((1, self.top_k), dtype=float)
                }

            # Extract prompt_logprobs if available
            prompt_logprobs_list = getattr(choice, 'prompt_logprobs', None)

            if prompt_logprobs_list is not None:
                n = len(prompt_logprobs_list)
                k = self.top_k

                # Initialize matrices for top-K indices and logprobs
                topk_indices = np.zeros((n, k), dtype=int)
                topk_logprobs = np.zeros((n, k), dtype=float)

                for i, logprob_dict in enumerate(prompt_logprobs_list):
                    if logprob_dict is None:
                        continue

                    # Sort items by logprob descending to ensure we get the top ones
                    # vLLM usually returns them sorted, but we sort to be safe
                    items = list(logprob_dict.items())
                    items.sort(key=lambda x: x[1]['logprob'] if isinstance(x[1], dict) else x[1].logprob, reverse=True)

                    for j, (token_id, info) in enumerate(items[:k]):
                        # token_id is expected to be the index (int or string representation of int)
                        try:
                            topk_indices[i, j] = int(token_id)
                        except (ValueError, TypeError):
                            pass

                        if isinstance(info, dict):
                            topk_logprobs[i, j] = info['logprob']
                        else:
                            topk_logprobs[i, j] = info.logprob

                # Original tokens and their logprobs for the actual sequence
                all_token_logprobs = choice.logprobs.token_logprobs
                all_tokens = choice.logprobs.tokens

                if len(all_token_logprobs) > 0:
                    all_token_logprobs[0] = 0.0

                return {
                    "prompt": all_tokens[:-1],
                    "prompt_logprobs": all_token_logprobs[:-1],
                    "topk_indices": topk_indices,
                    "topk_logprobs": topk_logprobs
                }
            else:
                # Fallback to original behavior if prompt_logprobs is not available
                all_token_logprobs = choice.logprobs.token_logprobs
                all_tokens = choice.logprobs.tokens

                if len(all_token_logprobs) > 0:
                    all_token_logprobs[0] = 0.0

                prompt_tokens = all_tokens[:-1]
                n = len(prompt_tokens)
                return {
                    "prompt": prompt_tokens,
                    "prompt_logprobs": all_token_logprobs[:-1],
                    "topk_indices": np.zeros((n, self.top_k), dtype=int),
                    "topk_logprobs": np.zeros((n, self.top_k), dtype=float)
                }

        except Exception as e:
            print(f"Error in _call_api_single: {e}")
            return {
                "prompt": [f"Error: {str(e)}"],
                "prompt_logprobs": [0.0],
                "topk_indices": np.zeros((1, self.top_k), dtype=int),
                "topk_logprobs": np.zeros((1, self.top_k), dtype=float)
            }
        
def load_data_from_jsonl(file_path: str) -> List[Dict]:

    data_list = []
    print(f"正在读取文件: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    item = json.loads(line)
                    if 'input' in item: data_list.append(item)
                except: pass
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
        return []
    print(f"成功加载 {len(data_list)} 条有效数据")
    return data_list

def main():
    # 1. 加载测试数据
    input_file_path = '/your/local/path'
    raw_data = load_data_from_jsonl(input_file_path)

    exp_n = 1

    if not raw_data:
        print("未找到测试数据，请检查路径。")
        return

    # 2. 提取测试 Prompt
    prompts = [item['input'] for item in raw_data][:exp_n]
    
    # 3. 初始化接口
    print("\n=== 开始测试 prompt_logprobs 返回功能 ===")
    api = vllmAPIModelInterface(model_name="Qwen3-32B")
    
    # 4. 调用接口
    start_time = time.time()
    results = api.get_batch_answers(prompts)
    elapsed_time = time.time() - start_time
    
    print(f"推理耗时: {elapsed_time:.4f} 秒\n")

    # 5. 详细验证
    for i, res in enumerate(results):
        print(f"--- 样本 {i+1} 验证 ---")
        if isinstance(res, dict) and "prompt_logprobs" in res:
            tokens = res.get("prompt", [])
            logprobs = res.get("prompt_logprobs", [])

            print(f"Prompt Token 数量: {len(tokens)}")
            print(f"Logprobs 数量: {len(logprobs)}")

            if len(logprobs) > 0:
                # 打印前 exp_n 个示例
                sample_len = min(exp_n, len(logprobs))
                logprobs[0] = 0.0

                # print(logprobs)

                # 计算平均概率
                avg_lp = sum(logprobs) / len(logprobs)
                print(f"平均 Logprob: {avg_lp:.4f}")

                if "topk_indices" in res:
                    print(f"Top-K Indices 形状: {res['topk_indices'].shape}")
                    print(f"Top-K Logprobs 形状: {res['topk_logprobs'].shape}")
                    print(f"前 5 个 Token 的前 3 个 Top-K 索引:\n{res['topk_indices'][:5, :3]}")
                    print(f"前 5 个 Token 的前 3 个 Top-K logprob:\n{res['topk_logprobs'][:5, :3]}")

                print("状态: [正常]")
            else:
                print("状态: [异常] logprobs 为空")
        else:
            print(f"状态: [失败] 返回格式异常: {res}")
        print("-" * 20)
        
if __name__ == "__main__":
    main()
    # client = OpenAI(
    #         base_url="http://127.0.0.1:8000/v1",
    #         api_key='your_api_key'
    #     )

    # response = client.completions.create(
    #                 model="Qwen3-8B",
    #                 prompt=[2,3],
    #                 logprobs=300,
    #                 max_tokens=1,
    #                 echo=True,
    #             )
    
    # choice = response.choices[0]
    # plpb = choice.prompt_logprobs[1]

    # for k, v in plpb.items():
    #     print(f"{type(k)}")
    #     print(f"{v['logprob']}")

    # print(choice.prompt_logprobs[1])