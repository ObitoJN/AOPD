# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from argparse import Namespace
from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import register, Dispatch
from verl.utils.api_interface import vllmAPIModelInterface
from transformers import AutoTokenizer

class vLLMScorerWorker(Worker):
    """
    外置 vLLM Client 模式的教师模型评分器。
    该 Worker 负责与外部 vLLM 服务通信，获取教师模型的 logprobs。
    """
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.role = kwargs.get('role')

        # 加载 Tokenizer 用于 Tensor -> Text 转换
        # 优先从 actor_rollout_ref.model.path 获取，确保与学生模型一致
        model_path = self.config.get('model', {}).get('path', self.config.actor_rollout_ref.model.path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        self.client = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """
        初始化外置 Client。
        """
        # 从配置中读取 API 参数
        # 兼容 distill.ip_pool 或 distill.server_url 等结构

        ip_pool = self.config.distill.ip_pool
        model_name = self.config.distill.model_name
        self.top_k = self.config.distill.top_k
        temperature = self.config.actor_rollout_ref.rollout.temperature
 
        self.client = vllmAPIModelInterface(
            model_name=model_name,
            ip_pool=ip_pool,
            top_k=self.top_k,
            temperature=temperature
        )

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_logprobs(self, data: DataProto) -> DataProto:
        """
        核心评分逻辑：提取 Response 部分的教师 logprobs。

        返回数据包含：
        - teacher_log_probs: (B, response_length) 每个 token 的对数概率
        - teacher_topk_logprobs: (B, response_length, top_k) 教师 top-k token 的对数概率
        - teacher_topk_indices: (B, response_length, top_k) 教师 top-k token 的 ID
        """
        input_ids = data.batch['input_ids'] # (B, seq_len)
        attention_mask = data.batch['attention_mask'] # (B, seq_len)
        response_length = data.batch['responses'].shape[1] # 从 responses 获取长度

        batch_size = input_ids.shape[0]
        top_k = self.top_k

        # 将输入转换为列表，移除 padding
        input_ids_list = [seq[mask==1].tolist() for seq, mask in zip(input_ids, attention_mask)]

        # 从外部 vLLM API 获取教师模型的 logprobs
        results = self.client.get_batch_answers(input_ids_list)

        # 初始化结果张量
        teacher_log_probs_res = torch.zeros((batch_size, response_length), dtype=torch.float32, device=input_ids.device)

        if top_k > 0:
            teacher_topk_indice_res = torch.zeros((batch_size, response_length, top_k), dtype=torch.long, device=input_ids.device)
            teacher_topk_logprobs_res = torch.zeros((batch_size, response_length, top_k), dtype=torch.float32, device=input_ids.device)
        else:
            teacher_topk_indice_res = None
            teacher_topk_logprobs_res = None

        for i, res in enumerate(results):
            teacher_logprobs = res.get('prompt_logprobs', None)
            topk_indices = res.get('topk_indices', None)
            topk_logprobs = res.get('topk_logprobs', None)

            # 检查返回数据的有效性
            if teacher_logprobs is None or len(teacher_logprobs) == 0:
                raise RuntimeError(
                    f"Teacher API 返回无效数据。样本索引: {i}，"
                    f"prompt_logprobs 为 None 或为空。"
                    f"这可能是因为 API 服务异常或返回了错误响应。"
                    f"请检查 Teacher API 服务状态。"
                )

            # 如果配置了 top_k，检查 top-k 数据的有效性
            if top_k > 0:
                if topk_indices is None or topk_logprobs is None:
                    raise RuntimeError(
                        f"Teacher API 返回无效的 top-k 数据。样本索引: {i}，"
                        f"配置了 top_k={top_k}，但 topk_indices 或 topk_logprobs 为 None。"
                        f"请检查 Teacher API 是否正确配置了 logprobs 参数。"
                    )

            # 计算响应部分的 mask（最后 response_length 个 token）
            res_mask = attention_mask[i, -response_length:]
            actual_res_token_num = int(res_mask.sum().item())

            if actual_res_token_num > 0:
                res_valid_indices = torch.nonzero(res_mask, as_tuple=True)[0]

                if teacher_logprobs is not None:
                    # 从教师 logprobs 中提取响应部分
                    # teacher_logprobs 是完整序列的 logprobs，我们只需要最后 actual_res_token_num 个
                    res_teacher_logprobs = teacher_logprobs[-actual_res_token_num:]
                    # 直接从 list/array 转换为 tensor
                    if isinstance(res_teacher_logprobs, list):
                        res_teacher_logprobs_tensor = torch.tensor(res_teacher_logprobs, dtype=torch.float32, device=input_ids.device)
                    else:
                        res_teacher_logprobs_tensor = torch.from_numpy(res_teacher_logprobs).to(device=input_ids.device, dtype=torch.float32)
                    # 填充到结果张量中（只填充有效的 token 位置）
                    teacher_log_probs_res[i, res_valid_indices] = res_teacher_logprobs_tensor

                # Handle top-K
                if top_k > 0 and topk_indices is not None and topk_logprobs is not None:
                    res_topk_indices = topk_indices[-actual_res_token_num:]
                    res_topk_logprobs = topk_logprobs[-actual_res_token_num:]

                    # 直接从 list/array 转换为 tensor
                    if isinstance(res_topk_indices, list):
                        indices_tensor = torch.tensor(res_topk_indices, dtype=torch.long, device=input_ids.device)
                    else:
                        indices_tensor = torch.from_numpy(res_topk_indices).to(device=input_ids.device, dtype=torch.long)

                    if isinstance(res_topk_logprobs, list):
                        logprobs_tensor = torch.tensor(res_topk_logprobs, dtype=torch.float32, device=input_ids.device)
                    else:
                        logprobs_tensor = torch.from_numpy(res_topk_logprobs).to(device=input_ids.device, dtype=torch.float32)

                    teacher_topk_indice_res[i, res_valid_indices] = indices_tensor
                    teacher_topk_logprobs_res[i, res_valid_indices] = logprobs_tensor

        # ===== 构建返回的 DataProto =====
        tensors = {
            'teacher_log_probs': teacher_log_probs_res
        }

        if teacher_topk_indice_res is not None:
            tensors['teacher_topk_indices'] = teacher_topk_indice_res
            tensors['teacher_topk_logprobs'] = teacher_topk_logprobs_res

        return DataProto.from_dict(tensors=tensors).to('cpu')



def create_mock_config():
    """
    创建模拟配置对象用于测试
    """
    config = Namespace()

    # 创建一个支持 .get() 方法的配置类
    class ConfigDict:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def get(self, key, default=None):
            return getattr(self, key, default)

    config.distill = ConfigDict(
        ip_pool=[],  # specify teacher server IPs, e.g. ['192.168.1.1', '192.168.1.2']
        model_name='Qwen3-32B',  # 示例模型名
        top_k=32
    )
    config.model = {'path': '/your/local/path'}
    config.actor_rollout_ref = Namespace(
        model=Namespace(path='/your/local/path')
    )
    return config


def test_vllm_client():
    """
    测试 vLLM API Client 的 get_batch_answers 结果
    """
    # 1. 创建 vLLM Client
    print("\n[步骤 1] 初始化 vLLM API Client...")
    ip_pool = []  # specify teacher server IPs here
    model_name = 'Qwen3-32B'
    top_k = 32

    try:
        client = vllmAPIModelInterface(
            model_name=model_name,
            ip_pool=ip_pool,
            top_k=top_k
        )
        print(f"✓ Client 初始化成功")
    except Exception as e:
        print(f"✗ Client 初始化失败: {e}")
        return

    # 2. 准备模拟输入数据
    print("\n[步骤 2] 准备模拟输入数据...")

    # 模拟 tokenizer - 这里使用简单的数字列表代表 token_ids
    mock_input_ids_list = [
        [1, 2, 3, 4, 5, 6, 7, 8],  # 样本 1
        [10, 20, 30, 40, 50],       # 样本 2
        [100, 200, 300, 400, 500, 600]  # 样本 3
    ]

    batch_size = len(mock_input_ids_list)

    try:
        results = client.get_batch_answers(mock_input_ids_list)
        print(f"✓ 成功获取 {len(results)} 个结果")
    except Exception as e:
        print(f"✗ 调用失败: {e}")
        return

    # 3. 处理结果
    print("\n[步骤 3] 处理返回结果...")
    for i, res in enumerate(results):
        all_valid_lps = res.get('prompt_logprobs', [])
        topk_indices = res.get('topk_indices', None)
        topk_logprobs = res.get('topk_logprobs', None)
        print(f"  样本 {i}: topk_indices shape={topk_indices.shape if topk_indices is not None else None}, "
              f"topk_logprobs shape={topk_logprobs.shape if topk_logprobs is not None else None}")

    return results


if __name__ == "__main__":
    # 运行测试
    results = test_vllm_client()
