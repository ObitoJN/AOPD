import json
import random
import numpy as np
from verl.utils.reward_score import math, gsm8k, math_verify, math_robust

def read_jsonl(file_path):
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            temp_json = json.loads(line)
            data_list.append(temp_json)
    
    return data_list

def write_jsonl(data_list, output_path):
    with open(output_path, 'w', encoding='utf-8') as file:
        for item in data_list:
            file.write(json.dumps(item, ensure_ascii=False))
            file.write('\n')
    
    print(f"文件生成至{output_path}.")

def compute_score(solution_str, ground_truth, data_source):
    if data_source == "openai/gsm8k":
        res = gsm8k.compute_score(solution_str, ground_truth, method='flexible')
        # res = math_verify.compute_score(solution_str, ground_truth)
        # res = math_robust.compute_score(solution_str, ground_truth)
    elif data_source == "math500":
        res = math_verify.compute_score(solution_str, ground_truth)
        # res = math_robust.compute_score(solution_str, ground_truth)
    elif data_source in ["aime2024", "aime2025", "hmmt_feb_2025", "minerva"]:
        # res = math_verify.compute_score(solution_str, ground_truth)
        res = math_robust.compute_score(solution_str, ground_truth)
    else:
        raise NotImplementedError(f"No support score function for datasource - {data_source}")

    return res

def single_eval(item, eval_mode="acc", at_num=None,data_source=None):
    '''
    eval_mode: options in ['acc', 'pass@k', 'maj@k']
    '''

    if data_source is None:
        data_source = item['data_source']
    solution_str = item['generated']
    ground_truth = item['answer']
    total_solution_list = [solution_str]
    if len(item["candidate_generated"]) > 0:
        candidate_solutions = item["candidate_generated"]
        total_solution_list = candidate_solutions
    
    # print(solution_str)
    
    candidate_solution_num = len(total_solution_list)
    
    if at_num != None:
        assert candidate_solution_num >= at_num, f"'Candidate solution num' {candidate_solution_num} should be equal to or larger than 'at_num' {at_num}."
    else:
        at_num = candidate_solution_num

    if eval_mode == "acc":
        res = compute_score(
                solution_str=total_solution_list[0], 
                ground_truth=ground_truth,
                data_source=data_source,
            )
        
    elif eval_mode == "pass@k":
        temp_score_list = []
        
        for solution in total_solution_list:
            temp_score = compute_score(
                    solution_str=solution, 
                    ground_truth=ground_truth,
                    data_source=data_source,
                )
            temp_score_list.append(temp_score)

        # Calculate pass@k using the unbiased estimator
        # pass@k = 1 - C(n-c, k) / C(n, k) where n=total samples, c=correct samples, k=samples to draw
        # This estimates the probability that at least one correct solution exists in k random samples
        n = len(temp_score_list)  # total number of samples
        c = sum(temp_score_list)   # number of correct samples
    
        if n < at_num:
            # Not enough samples
            res = 0.0
        elif c == 0:
            # No correct samples, pass@k = 0
            res = 0.0
        elif n - c < at_num:
            # If n-c < k, we cannot choose k samples without including at least one correct
            # This means: # of incorrect samples < k, so pass@k = 1.0
            res = 1.0
        else:
            # General case: Use the formula 1 - C(n-c, k) / C(n, k)
            # This equals: 1 - prod((n-c-i)/(n-i) for i in range(k))
            prob = 1.0
            for i in range(at_num):
                prob *= (n - c - i) / (n - i)
            res = 1.0 - prob

    elif eval_mode == "maj@k":
        # TODO: implement maj@k
        pass
    else:
        raise NotImplementedError(f"Not implement eval mode {eval_mode}.")
    
    return res, data_source, at_num

def eval_result(data, eval_mode="acc", at_num=None, print_avg=False, data_source=None):
    result_dict = {}
    for item in data:
        score, data_source, real_at_num = single_eval(item, eval_mode=eval_mode, at_num=at_num,data_source=data_source)
        if data_source not in result_dict:
            result_dict[data_source] = {
                "score": score,
                "at_num": real_at_num,
                "count": 1,
            }
        else:
            result_dict[data_source]["score"] += score
            result_dict[data_source]["count"] += 1
    
    total_count = 0
    sum_score = 0
    macro_score = 0
    total_data_source = len(result_dict)
    output_scores = {}

    for data_source, result in result_dict.items():
        sum_score += result["score"]
        total_count += result["count"]
        single_score = result["score"] / result["count"] 
        macro_score += single_score
        real_eval_mode = eval_mode.replace('k', str(result['at_num']))
        print(f"[{data_source}]--{real_eval_mode}: {single_score:.4f}")
        output_scores[real_eval_mode] = round(single_score, 4)
    
    if print_avg:
        print(f"Micro Score: {(sum_score / total_count):.4f}")
        print(f"Macro Score: {(macro_score / total_data_source):.4f}")

    return output_scores

if __name__ == "__main__":
    RESULT_PATH = "/your/local/path"
    
    MODEL_DICT = {
        "your/model/path"
            }

    # 数据集配置
    # 格式: {数据集名称: 分割数}
    # 总任务数 = 模型数 × 总分割数
    DATASET_DICT = {
        "aime2024": 6,
        "aime2025": 6,
        "hmmt_feb_2025": 1,
    }
    rollout=16


    eval_list = []
    all_results = {}

    max_model_len=32768

    for k, v in MODEL_DICT.items():
        print(f"\n===== {k} =====\n")
        all_results[k] = {}
        for dataset, split_num in DATASET_DICT.items():
            if 'aime' in dataset:
                max_token=38912
            else:
                max_token=32768

            max_token = min(max_token, max_model_len)
            RESULT_JSON = [f"{RESULT_PATH}/{dataset}_{k}_len{max_token}_t0.6_r{rollout}_{i}_result.json" for i in range(split_num)]
            
            total_data_list = []
            
            for file_path in RESULT_JSON:
                part_data = read_jsonl(file_path)
                total_data_list.extend(part_data)

            print(f"Dataset: {dataset}")
            # 评测pass@1，从dataset_dict的key作为data_source传入
            scores_1 = eval_result(total_data_list, eval_mode="pass@k", at_num=1, data_source=dataset)
            # 评测pass@k
            scores_4 = eval_result(total_data_list, eval_mode="pass@k", at_num=4, data_source=dataset)
            scores_8 = eval_result(total_data_list, eval_mode="pass@k", at_num=8, data_source=dataset)
            scores_16 = eval_result(total_data_list, eval_mode="pass@k", at_num=16, data_source=dataset)
            all_results[k][dataset] = {}
            all_results[k][dataset].update(scores_1)
            all_results[k][dataset].update(scores_4)
            all_results[k][dataset].update(scores_8)
            all_results[k][dataset].update(scores_16)
        print("\n")
    # 将评测结果写入文件
    output_file = f"{RESULT_PATH}/eval_summary.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n评测结果已保存至 {output_file}")
