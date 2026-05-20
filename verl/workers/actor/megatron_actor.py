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
"""
Megatron Actor.
In megatron actor, the differences are:
1. We only make minibatch

Note that our model doesn't have to be `MegatronModule` because we don't share embedding in the last layer
"""

import logging
import os
from functools import partial
from typing import Dict, Iterable

import torch
import torch.distributed
from megatron.core import parallel_state as mpu
from megatron.core.distributed import finalize_model_grads

# from megatron.core.optimizer import DistributedOptimizer
from megatron.core.optimizer import DistributedOptimizer
from megatron.core.pipeline_parallel import get_forward_backward_func
from omegaconf import OmegaConf
from torch import nn

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.debug.profile import Profiler
from verl.utils.megatron.pipeline_parallel import make_batch_generator
from verl.utils.megatron.tensor_parallel import vocab_parallel_entropy, vocab_parallel_log_probs_from_logits
from verl.utils.megatron_utils import get_model_config
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import broadcast_dict_tensor, split_dict_tensor_into_batches
from verl.workers.actor import BasePPOActor

__all__ = ["MegatronPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def compute_gkd_loss(
    data: dict,
    response_mask: torch.Tensor,
    logits: torch.Tensor = None,
    gkd_weight: float = 1.0,
    gkd_mask: torch.Tensor = None,
    jsd_beta: float = 0.5,
) -> torch.Tensor:
    """
    计算 GKD 损失（教师 top-K KL 散度或 JSD 散度）。

    支持 KL 散度和 JSD（Jensen-Shannon Divergence）两种损失类型：
    - 当 jsd_beta=1: 使用前向 KL 散度 KL(teacher_topk_probs || student_logps)
    - 当 jsd_beta=0: 使用反向 KL 散度 KL(student_topk_probs || teacher_topk_logps)
    - 当 0<jsd_beta<1: 使用 JSD 散度 = beta * KL(teacher||student) + (1-beta) * KL(student||teacher)

    正确处理 Megatron 的 Tensor Parallel (TP)，logits 可能是 vocab_parallel 的。

    Args:
        data: 包含 teacher_topk_logprobs, teacher_topk_indices 的字典
        response_mask: (batch_size, response_len) 响应掩码
        logits: (batch_size, response_len, vocab_size // tp_size) vocab_parallel logits
        gkd_weight: GKD 损失权重
        gkd_mask: (batch_size, response_len) GKD 区域掩码，仅在 aopd 模式下使用。为 None 则对所有 token 计算 GKD 损失
        jsd_beta: JSD 的 beta 参数，控制前向/反向 KL 的权重
                  - beta=1: 前向 KL（teacher||student）
                  - beta=0: 反向 KL（student||teacher）
                  - 0<beta<1: JSD 混合

    Returns:
        gkd_loss_tensor: (batch_size, response_len) GKD 损失张量
    """
    if "teacher_topk_logprobs" not in data or "teacher_topk_indices" not in data:
        return None

    if logits is None:
        return None

    teacher_topk_logps = data["teacher_topk_logprobs"]  # (batch_size, response_len, K)
    teacher_topk_indices = data["teacher_topk_indices"]  # (batch_size, response_len, K)

    response_len = response_mask.shape[1]
    batch_size = response_mask.shape[0]

    teacher_topk_probs = torch.exp(teacher_topk_logps)

    # 正确处理 vocab_parallel logits
    # 参考 recipe/gkd/megatron_kl_loss.py 的实现
    from megatron.core.fusions.fused_cross_entropy import calculate_logits_max
    from megatron.core.parallel_state import (
        get_tensor_model_parallel_group,
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )
    from megatron.core.tensor_parallel.utils import VocabUtility

    # 步骤 1: 计算 logits_max 并跨 TP ranks all_reduce
    vocab_parallel_logits, logits_max = calculate_logits_max(logits)
    partition_vocab_size = vocab_parallel_logits.size(-1)

    torch.distributed.all_reduce(
        logits_max, op=torch.distributed.ReduceOp.MAX, group=get_tensor_model_parallel_group()
    )

    # 步骤 2: 减去 max 并计算 exp
    vocab_parallel_logits = vocab_parallel_logits - logits_max.unsqueeze(dim=-1)
    vocab_parallel_logits = vocab_parallel_logits.exp_()
    exp_logits = vocab_parallel_logits

    # 步骤 3: 计算 sum_exp_logits 并跨 TP ranks all_reduce
    sum_exp_logits = exp_logits.sum(dim=-1)
    torch.distributed.all_reduce(
        sum_exp_logits,
        op=torch.distributed.ReduceOp.SUM,
        group=get_tensor_model_parallel_group(),
    )

    # 步骤 4: 计算当前 TP rank 的 vocab 范围
    rank = get_tensor_model_parallel_rank()
    world_size = get_tensor_model_parallel_world_size()
    vocab_start_index, vocab_end_index = VocabUtility.vocab_range_from_per_partition_vocab_size(
        partition_vocab_size, rank, world_size
    )

    # 步骤 5: 确定哪些 teacher top-k indices 在当前 partition 中
    topk_indices_in_vocab_mask = (teacher_topk_indices >= vocab_start_index) & (
        teacher_topk_indices < vocab_end_index
    )

    # 步骤 6: 将全局 indices 转换为当前 partition 的局部 indices
    vocab_parallel_topk_indices = teacher_topk_indices - vocab_start_index
    vocab_parallel_topk_indices = torch.where(
        topk_indices_in_vocab_mask,
        vocab_parallel_topk_indices,
        torch.zeros_like(vocab_parallel_topk_indices)
    )

    # 步骤 7: 计算 vocab_parallel 的概率分布
    vocab_parallel_probs = exp_logits / sum_exp_logits.unsqueeze(-1)
    vocab_parallel_probs_2d = vocab_parallel_probs.view(-1, partition_vocab_size)  # (batch*response_len, vocab_size//tp)

    # 步骤 8: 提取 top-k 位置的概率
    topk = teacher_topk_indices.size(-1)
    arange_1d = torch.arange(
        start=0, end=vocab_parallel_probs_2d.size(0), device=vocab_parallel_probs_2d.device
    )
    vocab_parallel_topk_probs_2d = vocab_parallel_probs_2d[
        arange_1d.unsqueeze(-1), vocab_parallel_topk_indices.view(-1, topk).long()
    ]  # (batch*response_len, K)
    vocab_parallel_topk_probs = vocab_parallel_topk_probs_2d.view(teacher_topk_indices.shape)  # (batch, response_len, K)

    # 步骤 9: 计算 log 概率，对不在当前 partition 的 token 设为 0
    vocab_parallel_topk_logps = torch.log(vocab_parallel_topk_probs + 1e-20)
    vocab_parallel_topk_logps = torch.where(
        topk_indices_in_vocab_mask,
        vocab_parallel_topk_logps,
        torch.zeros_like(vocab_parallel_topk_logps)
    )

    # 步骤 10: 对不在当前 partition 的 teacher probs 也设为 0（准备 all_reduce）
    vocab_parallel_teacher_topk_probs = torch.where(
        topk_indices_in_vocab_mask,
        teacher_topk_probs,
        torch.zeros_like(teacher_topk_probs)
    )
    vocab_parallel_teacher_topk_logps = torch.where(
        topk_indices_in_vocab_mask,
        teacher_topk_logps,
        torch.zeros_like(teacher_topk_logps)
    )

    # 步骤 11: 计算 KL 散度（每个 TP rank 只计算其 partition 的部分）
    # 前向 KL: KL(P||Q) = sum(P * (log(P) - log(Q)))
    forward_kl = torch.sum(
        vocab_parallel_teacher_topk_probs * (vocab_parallel_teacher_topk_logps - vocab_parallel_topk_logps),
        dim=-1
    )  # (batch_size, response_len)

    # 反向 KL: KL(Q||P) = sum(Q * (log(Q) - log(P)))
    reverse_kl = torch.sum(
        vocab_parallel_topk_probs * (vocab_parallel_topk_logps - vocab_parallel_teacher_topk_logps),
        dim=-1
    )  # (batch_size, response_len)

    # 步骤 12: 根据 beta 参数计算 JSD 损失
    # JSD = beta * KL(teacher||student) + (1-beta) * KL(student||teacher)
    jsd_divergence_per_token = jsd_beta * forward_kl + (1 - jsd_beta) * reverse_kl

    # 步骤 13: 跨 TP ranks all_reduce 得到完整的 JSD 散度
    torch.distributed.all_reduce(
        jsd_divergence_per_token,
        op=torch.distributed.ReduceOp.SUM,
        group=get_tensor_model_parallel_group(),
    )

    if gkd_mask is not None:
        jsd_divergence_per_token = jsd_divergence_per_token * gkd_mask.float()

    gkd_loss_tensor = gkd_weight * jsd_divergence_per_token
    gkd_loss_tensor = gkd_loss_tensor * response_mask.float()

    return gkd_loss_tensor


class MegatronPPOActor(BasePPOActor):
    def __init__(
        self,
        config,
        model_config,
        hf_config,
        tf_config,
        actor_module: nn.ModuleList,
        actor_optimizer: DistributedOptimizer,
    ):
        """MeagtronPPOActor class. This class implements the simple PPO logics when the model is built with Megatron.

        Args:
            config (OmegaConf): the basic config that contains the hyper-parameters of PPO Actor. It must contain

                ``ppo_micro_batch_size_per_gpu``: micro batch size when updating ppo.

                ``ppo_mini_batch_size``: minibatch size when updating ppo using the batch data.

                ``ppo_epochs``: number of epochs to update the actor using the batch data.

                ``shuffle``: whether to shuffle the data after each ppo epoch.

                ``clip_ratio``: clip ratio of the ppo algorithm. See https://arxiv.org/abs/1707.06347.

                ``entropy_coeff``: entropy coefficient of the PPO loss. See https://arxiv.org/abs/1707.06347.
            model_config (OmegaConf): model configuration. It must contains ``model_config.vocab_size`` and
                ``model_config.hidden_size``
            hf_config (PretrainedConfig): huggingface config
            tf_config (TransformerConfig): mcore transformer config
            actor_module (nn.ModuleList): actor module is a ModuleList that contains a list of nn.Module in this pp stage.
                each nn.Module in this rank holds a vpp module chunk. See https://arxiv.org/pdf/2104.04473.pdf for more details.
                The actor module has some constraints to follow in order to use the updating logics implemented here

                1. It must implement unpad_input before any computation and pad_input after all the computation. Remove padding is an
                optimization that removes the padding tokens. See unpad_input and pad_input function in flash-attn
                (https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/bert_padding.py).

                2. Each pp stage must return the hidden state with the same shape [total_nnz, 1, hidden_size],
                where total_nnz is the number of valid tokens in this batch. If sequence parallel is enabled, the size
                of the hidden state is [total_nnz // tp, 1, hidden_size].
            actor_optimizer (DistributedOptimizer): currently, we only support DistributedOptimizer in Megatron. It implements
                zero1 optimizer that shards the optimizer state across dp ranks.

        >>> from megatron.training import get_model
        >>> from megatron.optimizer import get_megatron_optimizer
        >>> actor_module = get_model(megatron_actor_model_provider, wrap_with_ddp=True)
        >>> actor_module = nn.ModuleList(actor_module)
        >>> actor_optimizer = get_megatron_optimizer(actor_module)
        >>> actor = MegatronPPOActor(config=config,
        >>>                          model_config=actor_model_config,
        >>>                          hf_config=hf_config,
        >>>                          tf_config=tf_config,
        >>>                          actor_module=actor_module,
        >>>                          actor_optimizer=actor_optimizer)
        """
        super().__init__(config)
        self._validate_config(config)
        self.model_config = model_config
        self.hf_config = hf_config
        self.tf_config = tf_config
        self.actor_module = actor_module
        self.actor_optimizer: DistributedOptimizer = actor_optimizer
        self.prof = Profiler(self.config.profile)
        self.optimizer_step_args = OmegaConf.create(
            {
                "skip_grad": None,
                "overlap_dp_param_comm": False,
                "overlap_dp_grad_comm": False,
                "gradient_accumulation_steps": 1,
                "sequence_parallel": self.tf_config.sequence_parallel,
                "DDP_impl": "local",
                "layernorm_allreduce_bucket_threshold": 0,
                "pipeline_model_parallel_split_rank": None,
                "reduce_grads_use_alltoall": False,
            }
        )

        config = get_model_config(self.actor_module[0])
        print(config)
        config.finalize_model_grads_func = finalize_model_grads

    def _validate_config(self, config) -> None:
        """Validate config options not implemented for Megatron backend"""
        assert config.get("ulysses_sequence_parallel_size", 1) == 1
        if config.get("shuffle", False):
            assert config.data_loader_seed is not None, "If shuffle dataloader, seed must be manually set"
        if config.megatron.tensor_model_parallel_size == 1:
            print("[Warining] Because actor tp size == 1, set sp to False")
            config.megatron.sequence_parallel = False
        self.config = config

    @GPUMemoryLogger(role="megatron actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            DataProto: torch.Tensor: the log_prob tensor
        """
        data.batch = data.batch.contiguous()

        def compute_logprobs_fn(output, data):
            response = data["responses"]
            response_length = response.size(1)
            logits = output
            logits = logits[:, -response_length - 1 : -1].contiguous()
            log_probs = vocab_parallel_log_probs_from_logits(logits, response)
            return {"log_probs": log_probs}

        # We make recompute_old_log_prob by default here.
        # TODO (zhangchi.usc1992): actually, this function should only return log_prob and this logic should be handled by user outside
        recompute_old_log_prob = self.config.get("recompute_old_log_prob", True)

        entropys = torch.Tensor()
        if recompute_old_log_prob:
            select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
            batch = data.select(batch_keys=select_keys).batch
            input_ids = batch["input_ids"]
            batch_size = input_ids.size(0)
            response = batch["responses"]
            response_length = response.size(1)
            with torch.no_grad():
                output = self.forward_backward_batch(data, forward_only=True, post_process_fn=compute_logprobs_fn, calculate_entropy=calculate_entropy)
                if mpu.is_pipeline_last_stage(ignore_virtual=True):
                    # only on last rank. It should be on every tp rank
                    if calculate_entropy:
                        log_probs = torch.cat([o[0]["log_probs"] for o in output], dim=0)  # (bs, seq_size)
                    else:
                        log_probs = torch.cat([o["log_probs"] for o in output], dim=0)  # (bs, seq_size)
                    log_probs = log_probs.to(torch.float32)
                else:
                    log_probs = torch.empty(size=(batch_size, response_length), dtype=torch.float32, device=input_ids.device)

                # broadcast across pp ranks
                torch.distributed.broadcast(
                    tensor=log_probs,
                    src=mpu.get_pipeline_model_parallel_last_rank(),
                    group=mpu.get_pipeline_model_parallel_group(),
                    async_op=False,
                )
                if calculate_entropy:
                    # Note that o[0] is metrics, o[1] is entropy
                    if mpu.is_pipeline_last_stage(ignore_virtual=True):
                        entropys = torch.cat([o[1] for o in output], dim=0)
                        entropys = entropys.to(torch.float32)
                    else:
                        entropys = torch.empty(size=(batch_size, response_length), dtype=torch.float32, device=input_ids.device)
                    # broadcast across pp ranks
                    torch.distributed.broadcast(
                        tensor=entropys,
                        src=mpu.get_pipeline_model_parallel_last_rank(),
                        group=mpu.get_pipeline_model_parallel_group(),
                        async_op=False,
                    )

        # add empty cache after each compute
        torch.cuda.empty_cache()

        return log_probs, entropys

    def make_minibatch_iterator(self, data: DataProto) -> Iterable[DataProto]:
        """Make minibatch iterator for updating the actor

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64, where ``sequence_length = prompt_length + response_length``

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64

                ``responses``: tensor of shape [batch_size, response_length]. torch.int64. Note that responses = input_ids[:, -response_length:]

                ``old_log_probs``: tensor of shape [batch_size, response_length]. torch.float32. The log probability of responses.

                ``advantages``: tensor of shape [batch_size, response_length]. torch.float32. The advantages of responses.
                See PPO paper for details. https://arxiv.org/abs/1707.06347

        Returns:

        """
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # 添加用于蒸馏（OPD/GKD）损失计算的 teacher 相关数据
        if self.config.distill_signal_in_loss or "teacher_log_probs" in data.batch:
            select_keys.append("teacher_log_probs")
            if "teacher_topk_logprobs" in data.batch:
                select_keys.append("teacher_topk_logprobs")
            if "teacher_topk_indices" in data.batch:
                select_keys.append("teacher_topk_indices")
        data = data.select(batch_keys=select_keys)
        return data.make_iterator(
            mini_batch_size=self.config.ppo_mini_batch_size,
            epochs=self.config.ppo_epochs,
            seed=self.config.data_loader_seed,
            dataloader_kwargs={"shuffle": self.config.shuffle},
        )

    def forward_backward_batch(self, data: DataProto, forward_only=False, post_process_fn=None, calculate_entropy=False, distill_mode="aopd", gkd_weight=1.0, aopd_threshold=0.1, jsd_beta=0.5):
        """
        We assume:
        - The model takes input: (input_ids, attention_mask, position_ids). No rmpad for the input
        - The communication shape is (total_nnz_pad_to_sp // tp_size, 1, hidden_size) if sequence parallel is enabled
        """
        # broadcast from last pp rank to all other pp ranks
        # TODO: actually, we just need to control the sampling order.
        broadcast_dict_tensor(data.batch, src=mpu.get_pipeline_model_parallel_last_rank(), group=mpu.get_pipeline_model_parallel_group())
        # split into micro-batches
        data.batch["attention_mask"] = data.batch["attention_mask"].to(bool)

        if data.meta_info.get("micro_batch_size", None) is not None:
            batch_size = data.meta_info["micro_batch_size"]
        else:
            batch_size = self.config.ppo_micro_batch_size_per_gpu
        batches = split_dict_tensor_into_batches(data.batch, batch_size=batch_size)
        # compute input shapes for pp stages
        n_micro_batch = len(batches)
        seq_len = batches[0]["input_ids"].shape[1]

        forward_backward_func = get_forward_backward_func()

        def loss_func(output, data, meta_info):
            # For memory efficiency
            # We move calculation of entropy to compute_log_probs, forward_only == True
            metrics = {}
            if forward_only:
                if post_process_fn is None:
                    metrics["logits"] = output
                else:
                    stats = post_process_fn(output, data)
                    metrics.update(stats)
                if not calculate_entropy:
                    return torch.tensor(1.0, device=output.device), metrics

            responses = data["responses"]
            response_length = responses.size(1)
            attention_mask = data["attention_mask"]
            response_mask = attention_mask[:, -response_length:]
            loss_agg_mode = self.config.loss_agg_mode

            # compute policy loss
            logits = output
            logits = logits[:, -response_length - 1 : -1].contiguous()
            ret_entropy = None
            if not forward_only:
                old_log_prob = data["old_log_probs"]
                advantages = data["advantages"]

                clip_ratio = meta_info["clip_ratio"]
                clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                clip_ratio_c = meta_info["clip_ratio_c"]
                log_prob = vocab_parallel_log_probs_from_logits(logits, responses)
                pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                    old_log_prob=old_log_prob,
                    log_prob=log_prob,
                    advantages=advantages,
                    response_mask=response_mask,
                    cliprange=clip_ratio,
                    cliprange_low=clip_ratio_low,
                    cliprange_high=clip_ratio_high,
                    clip_ratio_c=clip_ratio_c,
                    loss_agg_mode=loss_agg_mode,
                )
                policy_loss = pg_loss
            if calculate_entropy:
                entropy = vocab_parallel_entropy(logits)
                if not forward_only:
                    entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                    entropy_coeff = meta_info["entropy_coeff"]
                    policy_loss = pg_loss - entropy_coeff * entropy_loss
                else:
                    ret_entropy = entropy

            stats = {}
            if forward_only:
                policy_loss = torch.tensor(1.0, device=output.device)
            else:
                if self.config.use_kl_loss:
                    ref_log_prob = data["ref_log_prob"]
                    # compute kl loss
                    kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                    kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=self.config.loss_agg_mode)

                    policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                    metrics["actor/kl_loss"] = kl_loss.detach().item()
                    metrics["actor/kl_coef"] = self.config.kl_loss_coef

                # 处理 OPD (逆向 KL) 损失（当 distill_signal_in_loss=True 时）
                # 注意：当把 KL 当作奖励信号时（用于 compute_advantage），不能选择 distill_loss_type
                if self.config.distill_signal_in_loss:
                    if "teacher_log_probs" in data:
                        student_logprob = log_prob
                        teacher_logprob = data["teacher_log_probs"]
                        distill_kl_type = self.config.distill_loss_type
                        distill_kld = kl_penalty(student_logprob, teacher_logprob, kl_penalty=distill_kl_type)
                        distill_loss = agg_loss(loss_mat=distill_kld, loss_mask=response_mask, loss_agg_mode=self.config.loss_agg_mode)
                        policy_loss = policy_loss + distill_loss
                        metrics["actor/distill_loss"] = distill_loss.detach().item()

                # 处理 GKD 损失（在损失计算阶段）
                # GKD 是一种受监督损失，直接引导学生模型学习教师的 top-k 分布
                # 仅在 hybrid 或 gkd 模式下计算 GKD 损失
                if distill_mode in ['aopd', 'gkd'] and "teacher_log_probs" in data:
                    # 从 meta_info 中获取 distill 参数
                    gkd_weight_local = meta_info.get("gkd_weight", 1.0) if meta_info else 1.0
                    gkd_mask_local = None
                    if distill_mode == 'aopd':
                        aopd_threshold_local = meta_info.get("aopd_threshold", 0.1) if meta_info else 0.1
                        # Convert log probabilities to probabilities for comparison
                        student_probs_local = torch.exp(data["old_log_probs"])
                        teacher_probs_local = torch.exp(data["teacher_log_probs"][:, -response_length:])
                        # GKD region: student_probs - teacher_probs > threshold
                        prob_diff_local = student_probs_local - teacher_probs_local
                        gkd_mask_local = prob_diff_local > aopd_threshold_local

                        # 记录 GKD token 比例（仅在 aopd 模式）
                        gkd_tokens = (gkd_mask_local.float() * response_mask).sum().item()
                        total_tokens = response_mask.sum().item()
                        gkd_ratio = gkd_tokens / total_tokens if total_tokens > 0 else 0.0
                        metrics["actor/gkd_token_ratio"] = gkd_ratio
                    # 传递 logits 给 compute_gkd_loss，logits 在第455行已经计算好了
                    jsd_beta_local = meta_info.get("jsd_beta", 0.5) if meta_info else 0.5  # 默认值为 0.5 (JSD)
                    gkd_loss_tensor = compute_gkd_loss(data, response_mask, logits=logits, gkd_weight=gkd_weight_local, gkd_mask=gkd_mask_local, jsd_beta=jsd_beta_local)
                    if gkd_loss_tensor is not None:
                        gkd_loss = agg_loss(loss_mat=gkd_loss_tensor, loss_mask=response_mask, loss_agg_mode=self.config.loss_agg_mode)
                        policy_loss = policy_loss + gkd_loss
                        metrics["actor/gkd_loss"] = gkd_loss.detach().item()

                # return loss and stats
                stats.update(
                    {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                    }
                )
            append_to_dict(metrics, stats)
            return policy_loss, [metrics, ret_entropy]

        def forward_step(batch_iter, model):
            batch = next(batch_iter)
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            position_ids = batch["position_ids"]
            from verl.models.mcore import get_mcore_forward_fn

            forward_fn = get_mcore_forward_fn(self.hf_config)

            output = forward_fn(model, input_ids, attention_mask, position_ids, sequence_parallel=self.tf_config.sequence_parallel)
            if forward_only:
                meta_info = None
            else:
                clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                meta_info = {
                    "clip_ratio": self.config.clip_ratio,
                    "entropy_coeff": self.config.entropy_coeff,
                    "clip_ratio_c": clip_ratio_c,
                    "distill_mode": distill_mode,
                    "gkd_weight": gkd_weight,
                    "aopd_threshold": aopd_threshold,
                    "jsd_beta": jsd_beta,
                }
            return output, partial(loss_func, data=batch, meta_info=meta_info)

        # batch should be a list of batches inside micro-batches
        batch_generator = make_batch_generator(batches, vpp_size=len(self.actor_module))

        # TODO: we may use the new schedule instead
        # for flash-attn: (seq_len, batch_size, hidden_size) = (mbs*seq_len, 1, hidden_size)
        if mpu.get_pipeline_model_parallel_world_size() > 1:
            losses_reduced = forward_backward_func(
                forward_step_func=forward_step,
                data_iterator=batch_generator,
                model=self.actor_module,
                num_microbatches=n_micro_batch,
                seq_length=batch_size * seq_len,  # no use when input_shapes was set
                micro_batch_size=1,  # no use when input_shapes was set
                forward_only=forward_only,
            )
        else:
            losses_reduced = forward_backward_func(
                forward_step_func=forward_step,
                data_iterator=batch_generator,
                model=self.actor_module,
                num_microbatches=n_micro_batch,
                seq_length=batch_size * seq_len,  # in use for pp = 1
                micro_batch_size=1,  # in use for pp = 1
                forward_only=forward_only,
            )
        # loss_reduces contains the stats returned from loss_func
        return losses_reduced

    @GPUMemoryLogger(role="megatron actor", logger=logger)
    def update_policy(self, dataloader: Iterable[DataProto]) -> Dict:
        """Update the policy with an iterator of DataProto

        Args:
            dataloader (Iterable[DataProto]): an iterator over the DataProto that returns by ``make_minibatch_iterator``
                The keys of each data batch is described in the make_minibatch_iterator.

        Returns:
            Dict: a dictionary containing the statistics. Note that the statistics are only valid in the last pp stage
            and users have to combine the output in each dp rank manually.

        """
        metrics = {}
        self.prof.start()
        for data in dataloader:
            # data = data.batch.to(self.actor_module.device)
            self.actor_optimizer.zero_grad()
            # use use_contiguous_buffers_in_local_ddp and no overlap_dp_param_comm
            for chunk in self.actor_module:
                # if use distributed optimizer, zero grad buffer will be handled by optimizer
                chunk.zero_grad_buffer()

            # 从 self.config 获取蒸馏参数
            distill_mode = getattr(self.config, "distill_mode", None)

            # 从 self.config.distill 获取蒸馏权重和阈值
            if distill_mode in ['aopd', 'gkd']:
                gkd_weight = self.config.distill.gkd_weight
                aopd_threshold = self.config.distill.aopd_threshold
                jsd_beta = getattr(self.config.distill, 'jsd_beta', 0.5)  # 默认值为 0.5 (JSD)
            else:
                gkd_weight = None
                aopd_threshold = None
                jsd_beta = None

            calculate_entropy = self.config.entropy_coeff != 0
            metric_micro_batch = self.forward_backward_batch(data, calculate_entropy=calculate_entropy, distill_mode=distill_mode, gkd_weight=gkd_weight, aopd_threshold=aopd_threshold, jsd_beta=jsd_beta)
            for metric in metric_micro_batch:
                # Note that o[0] is metrics, o[1] is entropy, o[2] is response_mask
                append_to_dict(metrics, metric[0])  # append the metric from this micro-batch to global metrics.

            update_successful, grad_norm, num_zeros_in_grad = self.actor_optimizer.step()

            if update_successful:
                # allgather already execute in optimizer.step in new megatron
                pass
            else:
                raise NotImplementedError
            self.prof.step()
        # add empty cache after each compute
        self.prof.stop_and_save()
        self.prof.stop_trace()
        torch.cuda.empty_cache()
        return metrics
