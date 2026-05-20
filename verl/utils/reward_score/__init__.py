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
# from . import gsm8k, math, prime_math, prime_code


def _default_compute_score(data_source, solution_str, ground_truth, extra_info=None):
    if data_source == "openai/gsm8k":
        from . import gsm8k

        res = gsm8k.compute_score(solution_str, ground_truth, method='flexible')
    
    elif data_source in ["deepmath"]:
        from . import math

        res = math.compute_score(solution_str, ground_truth)
    
    elif data_source in ["deepmath_verify"]:
        from . import math_verify

        res = math_verify.compute_score(solution_str, ground_truth)

    elif data_source in ['tulu3']:
        return 0

    elif data_source in ["math500"]:
        from . import math

        res = math.compute_score(solution_str, ground_truth)

        # res = math.compute_score(solution_str, ground_truth)
        # [Optional] Math-Verify Integration
        # For enhanced accuracy, consider utilizing Math-Verify (https://github.com/huggingface/Math-Verify).
        # Note: Math-Verify needs to be manually installed via pip: `pip install math-verify`.
        # To use it, override the `compute_score` function with the following implementation:

        # from . import math_verify
        # res = math_verify.compute_score(solution_str, ground_truth)
    elif data_source == "math_dapo" or data_source.startswith("aime"):
        from . import math_dapo

        res = math_dapo.compute_score(solution_str, ground_truth)

    elif data_source == "voc_merged_all":
        from . import voc_RMv3
        res = voc_RMv3.compute_score(solution_str, ground_truth)
    
    # elif data_source == "voc_merged":
    #     from . import voc_RMv1
    #     res = voc_RMv1.compute_score(solution_str, ground_truth)

    elif data_source in [
        "numina_aops_forum",
        "numina_synthetic_math",
        "numina_amc_aime",
        "numina_synthetic_amc",
        "numina_cn_k12",
        "numina_olympiads",
    ]:
        from . import prime_math

        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ["codecontests", "apps", "codeforces", "taco"]:
        from . import prime_code

        res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    elif data_source in ["hiyouga/geometry3k"]:
        from . import geo3k

        res = geo3k.compute_score(solution_str, ground_truth)
    elif data_source == "tool_alpaca":
        from verl.utils.metric.tool_alpaca_metrics import compute_tool_call_accuracy

        # Extract golden answer from extra_info
        golden_answer = []
        if extra_info and isinstance(extra_info, dict):
            golden_answer = extra_info.get("golden_answer", [])

        # Compute tool calling accuracy metrics
        metrics = compute_tool_call_accuracy(solution_str, golden_answer, fuzzy_match=True)

        # Return detailed metrics as dict
        res = {
            "score": metrics["tool_call_accuracy"],
            "tool_call_acc": metrics["tool_call_accuracy"],
            "action_match": 1.0 if metrics["action_match"] else 0.0,
            "param_match": 1.0 if metrics["param_match"] else 0.0,
            "parse_success": 1.0 if metrics["parsed_success"] else 0.0,
        }
    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        return res
    elif isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])
