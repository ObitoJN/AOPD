# Asymmetric On-Policy Distillation (AOPD)

This repository is the official implementation of the paper:
**[Asymmetric On-Policy Distillation: Bridging Exploitation and Imitation at the Token Level](https://arxiv.org/abs/2605.06387)**

Built on the [VERL](https://github.com/volcengine/verl) framework, this repository provides an open-source framework for large language model distillation based on Reinforcement Learning. It supports multiple distillation algorithms including Asymmetric On-Policy Distillation (AOPD), On-Policy Distillation (OPD), and Generalized Knowledge Distillation (GKD).


## Features
- **AOPD Distillation** (`aopd_distill`): Asymmetric on-policy distillation that bridges exploitation and imitation at the token level.
- **OPD Distillation** (`opd_distill`): Standard on-policy distillation using token-level teacher feedback.
- **GKD Distillation** (`gkd_distill`): Generalized Knowledge Distillation for language models.
- Distributed training backed by Ray and FSDP.
- High-throughput rollout using vLLM.

## Environment Setup

### Requirements
We used 24 A100-80G GPUs for training, with 16 GPUs allocated to training the student model and 8 GPUs used to deploy the teacher model.

Ensure you have Python 3.9+ and CUDA 12+ installed. 
Install the requirements (we recommend using a Conda environment):
```bash
conda create -n verl_distill python=3.10
conda activate verl_distill

# Install PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install vLLM and Ray
pip install vllm ray
```

Install the project:
```bash
pip install -e .
```

## Data Preparation

We use the following datasets for training:

| Task | Dataset | Link |
|------|---------|------|
| Warmup (SFT) | OpenThoughts | [open-thoughts/open-thoughts](https://github.com/open-thoughts/open-thoughts) |
| Math Reasoning | DeepMath | [zwhe99/DeepMath](https://github.com/zwhe99/DeepMath) |
| Tool Use | ToolAlpaca | [tangqiaoyu/ToolAlpaca](https://github.com/tangqiaoyu/ToolAlpaca) |


## Reproduction Workflow

The full reproduction pipeline consists of three stages: **warmup**, **teacher deployment**, and **distillation training**.

### Step 1: Warmup (SFT)
Before distillation, the student model should first be warmed up via supervised fine-tuning (SFT) on the [OpenThoughts](https://github.com/open-thoughts/open-thoughts) dataset. 
SFT is implemented via [LlamaFactory](https://github.com/hiyouga/LlamaFactory)

### Step 2: Deploy the Teacher Model
Deploy the teacher model as a vLLM inference service. The distillation scripts query the teacher online during training to obtain token-level feedback.

Configure `deploy_vllm.sh`:
- `model_name_or_path`: Path to the teacher model weights (HuggingFace format).
- `served_model_name`: The model name used by the API (e.g., `Qwen3-32B`).

Then launch:
```bash
bash deploy_vllm.sh
```

Once the service starts, note down the server IP address displayed in the terminal output.

### Step 3: Configure the Training Script
We provide ready-to-use launch scripts in the root directory:

| Script | Task | Method |
|--------|------|--------|
| `qwen3_8b_aopd_distill.sh` | Math | AOPD |
| `qwen3_8b_opd_distill.sh` | Math | OPD |
| `qwen3_8b_gkd_distill.sh` | Math | GKD |
| `tool_alpaca_aopd_distill.sh` | Tool Use | AOPD |
| `tool_alpaca_opd_distill.sh` | Tool Use | OPD |
| `tool_alpaca_gkd_distill.sh` | Tool Use | GKD |

Open your chosen script and configure the following variables:
- **`IP_POOL`**: Fill in the IP address(es) from Step 2 (e.g., `IP_POOL="['<teacher_ip_1>','<teacher_ip_2>']"`).
- **`POLICY_MODEL_PATH`**: Path to the warmed-up student model checkpoint.
- **`TRAIN_DATA`**: Path to your processed training `.parquet` file.
- **`VAL_DATA`**: Path to your processed validation `.parquet` file.
- **`SAVE_DIR`**: Directory where checkpoints and TensorBoard logs will be stored.

### Step 4: Launch Distillation
Run the training script from the project root directory on the master node:
```bash
bash qwen3_8b_aopd_distill.sh
```

The script will:
1. Start the Ray cluster (`ray start --head`).
2. Submit a Ray job that launches `verl/trainer/main_ppo.py`.
3. Save checkpoints and TensorBoard logs under `SAVE_DIR`.

For multi-node training, run the same script on all worker nodes. Each node auto-detects its role via the `NODE_RANK` environment variable (default: `0` for the head node).

## Evaluation
Evaluation scripts are provided under `scripts/eval/eval_benchmark.py`. Set `RESULT_PATH` and add your checkpoint paths to run evaluation on math reasoning benchmarks.

## Acknowledgements

We sincerely thank the following open-source projects that made this work possible:

- **[VERL](https://github.com/volcengine/verl)**: The RL training framework this codebase is built upon.
- **[vLLM](https://github.com/vllm-project/vllm)**: High-throughput inference engine used for teacher model serving and student rollout.
- **[OpenThoughts](https://github.com/open-thoughts/open-thoughts)**: Open dataset used for student model warmup.
- **[DeepMath](https://github.com/zwhe99/DeepMath)**: Mathematical reasoning dataset used for distillation training.
- **[ToolAlpaca](https://github.com/tangqiaoyu/ToolAlpaca)**: Tool-use dataset used for sequential tool-use adaptation experiments.

## Citation

If you find this work useful, please consider citing our paper:

```bibtex
@misc{jia2026asymmetriconpolicydistillationbridging,
      title={Asymmetric On-Policy Distillation: Bridging Exploitation and Imitation at the Token Level},
      author={Nan Jia and Haojin Yang and Xing Ma and Jiesong Lian and Shuailiang Zhang and Weipeng Zhang and Ke Zeng and Xunliang Cai and Zequn Sun},
      year={2026},
      eprint={2605.06387},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.06387},
}
