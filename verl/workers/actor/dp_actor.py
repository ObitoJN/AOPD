# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
Single Process Actor
"""

import itertools
import logging
import os
from typing import Tuple

import torch
from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor

__all__ = ["DataParallelPPOActor"]

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
                    注意：这是一个简化的 JSD，真正的 JSD 应该是对 teacher 和 student 的平均

    Args:
        data: 包含 teacher_topk_logprobs, teacher_topk_indices 的字典
        response_mask: (batch_size, response_len) 响应掩码
        logits: (batch_size, response_len, vocab_size) 学生模型的完整 logits
               **注意**: 这些 logits 必须已经应用了 temperature scaling (logits / temperature)
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

    teacher_topk_probs = torch.exp(teacher_topk_logps)
    student_log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # (batch_size, response_len, vocab_size)

    # 使用 gather 提取 teacher top-k 位置的学生 log 概率
    student_topk_logps = torch.gather(
        student_log_probs,
        dim=-1,
        index=teacher_topk_indices.long()
    )  # (batch_size, response_len, K)

    # 计算前向 KL 散度: KL(P||Q) = sum(P * (log(P) - log(Q)))
    # P = teacher_topk_probs, Q = student_topk_probs
    forward_kl = torch.sum(
        teacher_topk_probs * (teacher_topk_logps - student_topk_logps),
        dim=-1
    )  # (batch_size, response_len)

    # 计算反向 KL 散度: KL(Q||P) = sum(Q * (log(Q) - log(P)))
    # 注意：这里需要 student_topk_probs，即 exp(student_topk_logps)
    student_topk_probs = torch.exp(student_topk_logps)
    reverse_kl = torch.sum(
        student_topk_probs * (student_topk_logps - teacher_topk_logps),
        dim=-1
    )  # (batch_size, response_len)

    # 根据 beta 参数计算 JSD 损失
    # JSD = beta * KL(teacher||student) + (1-beta) * KL(student||teacher)
    # 注意：这不是标准的 JSD，标准 JSD 应该使用 (P+Q)/2 作为中间分布
    # 这里是为了灵活控制前向/反向 KL 的权重
    jsd_divergence_per_token = jsd_beta * forward_kl + (1 - jsd_beta) * reverse_kl

    if gkd_mask is not None:
        jsd_divergence_per_token = jsd_divergence_per_token * gkd_mask.float()

    gkd_loss_tensor = gkd_weight * jsd_divergence_per_token

    gkd_loss_tensor = gkd_loss_tensor * response_mask.float()

    return gkd_loss_tensor


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get("use_remove_padding", False)
        print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else verl_F.entropy_from_logits
        )

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False, return_logits=False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
            logits: # (bs, response_len, vocab_size) or None
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None, self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                logits_rmpad.div_(temperature)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                inplace_backward = True
                if calculate_entropy:
                    inplace_backward = False
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled, inplace_backward=inplace_backward)

                # compute entropy
                if calculate_entropy:
                    entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

                # For GKD loss computation with rmpad: we need to reconstruct logits
                # This is memory-intensive, so only do it when return_logits=True
                if return_logits:
                    # 需要 pad back logits_rmpad 以用于 GKD 损失计算
                    # 先 gather logits_rmpad（如果使用了 ulysses sp）
                    if self.use_ulysses_sp:
                        logits_rmpad_gathered = gather_outpus_and_unpad(logits_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    else:
                        logits_rmpad_gathered = logits_rmpad

                    # Pad back to (batch_size, seqlen, vocab_size)
                    vocab_size = logits_rmpad_gathered.size(-1)
                    full_logits = pad_input(
                        hidden_states=logits_rmpad_gathered.unsqueeze(1),  # (total_nnz, 1, vocab_size)
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen
                    )  # (batch_size, seqlen, 1, vocab_size)
                    full_logits = full_logits.squeeze(2)  # (batch_size, seqlen, vocab_size)

                    # 只返回 response 部分
                    logits_for_gkd = full_logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                else:
                    logits_for_gkd = None

            else:  # not using rmpad and no ulysses sp
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                if calculate_entropy:
                    entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

                # Store logits for GKD loss if requested
                logits_for_gkd = logits if return_logits else None

            return entropy, log_probs, logits_for_gkd

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
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
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs, _ = self._forward_micro_batch(micro_batch, temperature=temperature, calculate_entropy=calculate_entropy, return_logits=False)
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        multi_turn = data.meta_info.get("multi_turn", False)

        # 从 self.config 获取蒸馏模式
        distill_mode = getattr(self.config, "distill_mode", None)

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # 添加用于蒸馏（OPD/GKD）损失计算的 teacher 相关数据
        if self.config.distill_signal_in_loss or "teacher_log_probs" in data.batch:
            select_keys.append("teacher_log_probs")
            if "teacher_topk_logprobs" in data.batch:
                select_keys.append("teacher_topk_logprobs")
            if "teacher_topk_indices" in data.batch:
                select_keys.append("teacher_topk_indices")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(dataloader):
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch_idx, data in enumerate(micro_batches):
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(torch.cuda.current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(torch.cuda.current_device())  # actor device is cpu when using offload
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    if multi_turn:
                        response_mask = data["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    old_log_prob = data["old_log_probs"]
                    advantages = data["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True

                    # 检查是否需要返回 logits（用于 GKD 损失计算）
                    need_logits_for_gkd = distill_mode in ['aopd', 'gkd']
                    entropy, log_prob, logits_for_gkd = self._forward_micro_batch(
                        micro_batch=data,
                        temperature=temperature,
                        calculate_entropy=calculate_entropy,
                        return_logits=need_logits_for_gkd
                    )

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

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = data["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=self.config.loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if distill_mode in ['aopd', 'opd'] and self.config.distill_signal_in_loss:
                        # OPD (逆向 KL) 损失：使用 distill_loss_type
                        # 注意：当把 KL 当作奖励信号时（用于 compute_advantage），不能选择 distill_loss_type
                        student_logprob = log_prob
                        teacher_logprob = data["teacher_log_probs"]
                        distill_kl_type = self.config.distill_loss_type
                        distill_kld = kl_penalty(student_logprob, teacher_logprob, kl_penalty=distill_kl_type)
                        distill_loss = agg_loss(loss_mat=distill_kld, loss_mask=response_mask, loss_agg_mode=self.config.loss_agg_mode)
                        policy_loss = policy_loss + distill_loss
                        metrics["actor/distill_loss"] = distill_loss.detach().item()

                    # 处理 GKD 损失
                    # GKD 是一种监督损失，直接引导学生模型学习教师的 top-k 分布
                    if distill_mode in ['aopd', 'gkd']:
                        gkd_weight = self.config.distill.gkd_weight
                        jsd_beta = getattr(self.config.distill, 'jsd_beta', 0.5)  # 默认值为 0.5 (JSD)
                        gkd_mask_local = None
                        if distill_mode == 'aopd':
                            aopd_threshold = self.config.distill.aopd_threshold
                            # Convert log probabilities to probabilities for comparison
                            student_probs_local = torch.exp(data["old_log_probs"])
                            teacher_probs_local = torch.exp(data["teacher_log_probs"][:, -response_length:])
                            # GKD region: student_probs - teacher_probs > threshold
                            prob_diff_local = student_probs_local - teacher_probs_local
                            gkd_mask_local = prob_diff_local > aopd_threshold

                            # 记录 GKD token 比例（仅在 aopd 模式）
                            gkd_tokens = (gkd_mask_local.float() * response_mask).sum().item()
                            total_tokens = response_mask.sum().item()
                            gkd_ratio = gkd_tokens / total_tokens if total_tokens > 0 else 0.0
                            metrics["actor/gkd_token_ratio"] = gkd_ratio

                        gkd_loss_tensor = compute_gkd_loss(data, response_mask, logits=logits_for_gkd, gkd_weight=gkd_weight, gkd_mask=gkd_mask_local, jsd_beta=jsd_beta)
                        if gkd_loss_tensor is not None:
                            gkd_loss = agg_loss(loss_mat=gkd_loss_tensor, loss_mask=response_mask, loss_agg_mode=self.config.loss_agg_mode)
                            policy_loss = policy_loss + gkd_loss
                            metrics["actor/gkd_loss"] = gkd_loss.detach().item()

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    data = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                    }
                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
            append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics
