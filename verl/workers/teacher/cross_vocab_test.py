import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

class CrossTokenizerDistiller:
    def __init__(self, student_model_name, teacher_model_name):
        self.s_tokenizer = AutoTokenizer.from_pretrained(student_model_name, trust_remote_code=True)
        self.t_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name, trust_remote_code=True)
        if self.s_tokenizer.pad_token is None: self.s_tokenizer.pad_token = self.s_tokenizer.eos_token
        if self.t_tokenizer.pad_token is None: self.t_tokenizer.pad_token = self.t_tokenizer.eos_token

    def get_spans(self, tokenizer, token_ids):
        spans = []
        valid_ids = [tid for tid in token_ids if tid != tokenizer.pad_token_id]
        for k in range(len(valid_ids)):
            prev_text = tokenizer.decode(valid_ids[:k], skip_special_tokens=False)
            curr_text = tokenizer.decode(valid_ids[:k+1], skip_special_tokens=False)
            spans.append((len(prev_text), len(curr_text)))
        return spans, len(valid_ids)

    def align_batch(self, texts, teacher_log_probs):
        batch_size = len(texts)
        teacher_dim = teacher_log_probs.size(-1)
        s_inputs = self.s_tokenizer(texts, padding=True, return_tensors="pt", add_special_tokens=False)
        s_batch_ids = s_inputs["input_ids"]
        s_max_len = s_batch_ids.size(1)

        # 初始化输出：[B, S_max_len, D]
        aligned_batch_targets = torch.zeros(batch_size, s_max_len, teacher_dim, device=teacher_log_probs.device)

        for b in range(batch_size):
            text = texts[b]
            t_ids = self.t_tokenizer.encode(text, add_special_tokens=False)
            s_spans, s_valid_len = self.get_spans(self.s_tokenizer, s_batch_ids[b].tolist())
            t_spans, t_valid_len = self.get_spans(self.t_tokenizer, t_ids)

            # 映射逻辑：每个教师 j 对应哪个学生 i (取最后一个)
            t_to_s_map = []
            for j in range(t_valid_len):
                t_start, t_end = t_spans[j]
                last_i = -1
                for i in range(s_valid_len):
                    s_start, s_end = s_spans[i]
                    if max(s_start, t_start) < min(s_end, t_end):
                        last_i = i
                t_to_s_map.append(last_i)

            # 填充与累加
            for j in range(t_valid_len):
                target_i = t_to_s_map[j]
                if target_i != -1:
                    aligned_batch_targets[b, target_i, :] += teacher_log_probs[b, j, :]

            # 打印详细对照表
            self._visualize_sample(b, text, s_batch_ids[b].tolist(), t_ids, t_to_s_map, 
                                   s_valid_len, teacher_log_probs[b], aligned_batch_targets[b])

        return aligned_batch_targets

    def _visualize_sample(self, b_idx, text, s_ids, t_ids, t_to_s_map, s_valid_len, raw_t_probs, aligned_data):
        s_tokens = self.s_tokenizer.convert_ids_to_tokens(s_ids)
        t_tokens = self.t_tokenizer.convert_ids_to_tokens(t_ids)
        
        print(f"\n{'='*40} 样本 {b_idx} 对齐对照表 {'='*40}")
        print(f"文本: {text}")
        print(f"{'S_Idx':<5} | {'学生 Token':<15} | {'教师 Token (原始值)':<35} | {'对齐后 LogProb (Sum)':<20}")
        print("-" * 120)
        
        # 记录学生 i 接收了哪些教师 j
        s_receipt = [[] for _ in range(len(s_ids))]
        for j, i in enumerate(t_to_s_map):
            if i != -1: s_receipt[i].append(j)

        for i in range(len(s_ids)):
            is_pad = i >= s_valid_len
            matched_js = s_receipt[i]
            final_val = aligned_data[i, 0].item()
            
            if is_pad:
                t_info = "[PADDING]"
                sum_info = "-"
            elif not matched_js:
                t_info = "None"
                sum_info = f"{final_val:.1f} (Split)"
            else:
                # 展示合并过程：例如 "T0(0.0) + T1(1.0)"
                details = []
                for j in matched_js:
                    raw_val = raw_t_probs[j, 0].item()
                    details.append(f"T{j}({raw_val:.1f})")
                t_info = " + ".join(details)
                sum_info = f"{final_val:.1f}"
            
            print(f"{i:<5} | {s_tokens[i]:<15} | {t_info:<35} | {sum_info:<20}")
        print("-" * 120)

def run_test():
    student_path = "/your/local/path"
    teacher_path = "/your/local/path"
    
    batch_texts = [
        "友谊之花。", 
        "Amazing! 友谊之花。"
    ]

    distiller = CrossTokenizerDistiller(student_path, teacher_path)
    
    # 1. 准备教师 Batch 维度
    t_inputs = distiller.t_tokenizer(batch_texts, padding=True, return_tensors="pt", add_special_tokens=False)
    batch_size, t_max_len = t_inputs["input_ids"].shape
    teacher_dim = 1 
    
    # 2. 构造等差数列模拟数据
    # Batch 0: 0, 1, 2...
    # Batch 1: 100, 101, 102...
    mock_teacher_batch_log_probs = torch.zeros(batch_size, t_max_len, teacher_dim)
    for b in range(batch_size):
        for j in range(t_max_len):
            mock_teacher_batch_log_probs[b, j, :] = (b * 100.0) + float(j)

    # 3. 执行对齐
    distiller.align_batch(batch_texts, mock_teacher_batch_log_probs)

if __name__ == "__main__":
    run_test()