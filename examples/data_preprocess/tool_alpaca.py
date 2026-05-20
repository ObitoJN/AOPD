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
Preprocess the ToolAlpaca dataset to parquet format for distillation training with tool calling validation
"""

import argparse
import json
import os

from verl.utils.hdfs_io import copy, makedirs


def process_tool_alpaca_data(data_path, split):
    """Process ToolAlpaca data from JSONL file

    Convert ToolAlpaca format to training format suitable for:
    - Distillation training (蒸馏训练)
    - Tool calling validation metrics (验证指标)
    """

    def process_fn(line_data, idx):
        """Convert a single ToolAlpaca example to training format

        Input: ToolAlpaca format with:
          - prompt: full prompt string (includes tool info and instructions)
          - golden_answer: [{Action, Action_Input}]
          - golden_response: model response text

        Output: Training format with:
          - prompt: list of messages for chat template
          - extra_info: contains golden_answer for validation metrics
        """

        # Use the original prompt directly as user message
        # The prompt field already contains all tool information and instructions
        prompt_text = line_data.get("prompt", "")

        # Build chat message format (compatible with tokenizer.apply_chat_template)
        prompt_list = [
            {
                "role": "user",
                "content": prompt_text,
            },
        ]

        # Extract golden answers and responses
        golden_answer = line_data.get("golden_answer", [])
        golden_response = line_data.get("golden_response", [])

        data = {
            "data_source": "tool_alpaca",
            "prompt": prompt_list,
            "ability": "tool_use",
            "reward_model": {
                "style": "rule",
                "ground_truth": "",  # No explicit ground truth for distillation
            },
            "extra_info": {
                "split": split,
                "index": idx,
                # Store golden answer for validation metrics
                "golden_answer": golden_answer,
                "golden_response": golden_response,
            },
        }
        return data

    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_dir",
        default="~/data/tool_alpaca",
        help="Local directory to save preprocessed data",
    )
    parser.add_argument(
        "--hdfs_dir",
        default=None,
        help="HDFS directory to save preprocessed data",
    )
    parser.add_argument(
        "--train_file",
        default=None,
        help="Path to train JSONL file",
    )
    parser.add_argument(
        "--test_file",
        default=None,
        help="Path to test JSONL file",
    )

    args = parser.parse_args()

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    # Process training data
    if args.train_file:
        print(f"Processing training data from {args.train_file}...")
        train_data = []
        process_fn = process_tool_alpaca_data(args.train_file, "train")

        with open(args.train_file, "r", encoding="utf-8") as f:
            file_content = f.read().strip()
            # Try to parse as JSON array first
            if file_content.startswith("["):
                data_list = json.loads(file_content)
                if isinstance(data_list, list):
                    for idx, line_data in enumerate(data_list):
                        try:
                            processed = process_fn(line_data, idx)
                            train_data.append(processed)
                        except Exception as e:
                            print(f"Warning: Failed to process item {idx}: {e}")
                            continue
                else:
                    raise ValueError("File format is not a JSON array")
            else:
                # Fallback to JSONL format
                f.seek(0)
                for idx, line in enumerate(f):
                    try:
                        line_data = json.loads(line.strip())
                        processed = process_fn(line_data, idx)
                        train_data.append(processed)
                    except Exception as e:
                        print(f"Warning: Failed to process line {idx}: {e}")
                        continue

        import pandas as pd
        train_df = pd.DataFrame(train_data)
        train_path = os.path.join(local_dir, "train.parquet")
        train_df.to_parquet(train_path)
        print(f"Saved {len(train_data)} training samples to {train_path}")

    # Process test data
    if args.test_file:
        print(f"Processing test data from {args.test_file}...")
        test_data = []
        process_fn = process_tool_alpaca_data(args.test_file, "test")

        with open(args.test_file, "r", encoding="utf-8") as f:
            file_content = f.read().strip()
            # Try to parse as JSON array first
            if file_content.startswith("["):
                data_list = json.loads(file_content)
                if isinstance(data_list, list):
                    for idx, line_data in enumerate(data_list):
                        try:
                            processed = process_fn(line_data, idx)
                            test_data.append(processed)
                        except Exception as e:
                            print(f"Warning: Failed to process item {idx}: {e}")
                            continue
                else:
                    raise ValueError("File format is not a JSON array")
            else:
                # Fallback to JSONL format
                f.seek(0)
                for idx, line in enumerate(f):
                    try:
                        line_data = json.loads(line.strip())
                        processed = process_fn(line_data, idx)
                        test_data.append(processed)
                    except Exception as e:
                        print(f"Warning: Failed to process line {idx}: {e}")
                        continue

        import pandas as pd
        test_df = pd.DataFrame(test_data)
        test_path = os.path.join(local_dir, "test.parquet")
        test_df.to_parquet(test_path)
        print(f"Saved {len(test_data)} test samples to {test_path}")

    # Copy to HDFS if specified
    if args.hdfs_dir is not None:
        print(f"Copying to HDFS: {args.hdfs_dir}")
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
        print("HDFS copy completed")

    print("ToolAlpaca data preprocessing completed!")
