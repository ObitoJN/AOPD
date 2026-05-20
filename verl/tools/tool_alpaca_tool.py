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

import json
import logging
import os
from typing import Any, Optional, Tuple
from uuid import uuid4

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ToolAlpacaTool(BaseTool):
    """A tool for evaluating ToolAlpaca model outputs.

    This tool evaluates whether a model's tool call output matches the golden answer
    from the ToolAlpaca dataset. It supports checking:
    - Correct tool selection
    - Proper parameter formatting
    - Action/Action Input structure
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        Initialize ToolAlpacaTool.

        Args:
            config: Configuration dict (can contain evaluation settings)
            tool_schema: OpenAI format tool schema
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.use_fuzzy_matching = config.get("use_fuzzy_matching", True)
        self.debug = config.get("debug", False)

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(
        self, instance_id: Optional[str] = None, golden_answer: Optional[list] = None, **kwargs
    ) -> str:
        """Create a tool instance for a trajectory.

        Args:
            instance_id: The instance id of the tool
            golden_answer: The expected golden answer (list of tool call dicts)

        Returns:
            The instance id
        """
        if instance_id is None:
            instance_id = str(uuid4())

        self._instance_dict[instance_id] = {
            "golden_answer": golden_answer or [],
            "current_response": "",
            "reward": 0.0,
            "call_count": 0,
        }
        return instance_id

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> Tuple[str, float, dict]:
        """Execute the tool and evaluate the model's tool call.

        Args:
            instance_id: The instance id
            parameters: The tool call parameters (should contain "Action" and "Action Input")

        Returns:
            Tuple of (response_str, reward_score, metrics_dict)
        """
        if instance_id not in self._instance_dict:
            return "Tool not initialized", 0.0, {}

        self._instance_dict[instance_id]["call_count"] += 1
        call_count = self._instance_dict[instance_id]["call_count"]

        # Extract action and parameters from the model's output
        action = parameters.get("Action", "")
        action_input = parameters.get("Action Input", "")

        # Convert action_input to dict if it's a string
        if isinstance(action_input, str):
            try:
                action_input_dict = json.loads(action_input)
            except json.JSONDecodeError:
                action_input_dict = {}
        else:
            action_input_dict = action_input

        # Get the expected golden answer for this call
        golden_answers = self._instance_dict[instance_id]["golden_answer"]
        expected_action = None
        expected_input = None

        if call_count - 1 < len(golden_answers):
            expected_data = golden_answers[call_count - 1]
            expected_action = expected_data.get("Action", "")
            expected_input_str = expected_data.get("Action_Input", "")
            if isinstance(expected_input_str, str):
                try:
                    expected_input = json.loads(expected_input_str)
                except json.JSONDecodeError:
                    expected_input = {}
            else:
                expected_input = expected_input_str

        # Evaluate the tool call
        reward = 0.0
        metrics = {}

        if expected_action:
            # Check if action matches
            action_match = action.lower() == expected_action.lower()
            metrics["action_match"] = action_match

            if action_match:
                reward += 0.5

                # Check if action input matches
                if expected_input:
                    input_match = self._compare_inputs(action_input_dict, expected_input)
                    metrics["input_match"] = input_match
                    if input_match:
                        reward += 0.5
                    else:
                        if self.debug:
                            logger.warning(
                                f"Input mismatch for {action}: "
                                f"expected {expected_input}, got {action_input_dict}"
                            )
                else:
                    reward += 0.5
                    metrics["input_match"] = True
            else:
                if self.debug:
                    logger.warning(
                        f"Action mismatch: expected {expected_action}, got {action}"
                    )
        else:
            # No expected action for this call, give partial reward
            reward = 0.3
            metrics["action_match"] = None

        metrics["call_number"] = call_count
        self._instance_dict[instance_id]["reward"] = max(
            self._instance_dict[instance_id]["reward"], reward
        )

        response_msg = f"Tool call {call_count} evaluated. Action: {action}, Reward: {reward:.2f}"
        return response_msg, reward, metrics

    def _compare_inputs(self, actual: dict, expected: dict) -> bool:
        """Compare if actual input matches expected input.

        Args:
            actual: Actual input from model
            expected: Expected input from golden answer

        Returns:
            True if inputs match (with fuzzy matching if enabled)
        """
        if not self.use_fuzzy_matching:
            return actual == expected

        # Fuzzy matching: check if all expected keys are present in actual
        for key, value in expected.items():
            if key not in actual:
                return False

            # For string values, do case-insensitive comparison
            if isinstance(value, str) and isinstance(actual[key], str):
                if value.lower() != actual[key].lower():
                    return False
            elif value != actual[key]:
                return False

        return True

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate the final reward for this instance.

        Args:
            instance_id: The instance id

        Returns:
            The final reward score
        """
        if instance_id not in self._instance_dict:
            return 0.0

        instance_data = self._instance_dict[instance_id]
        golden_answers = instance_data["golden_answer"]
        call_count = instance_data["call_count"]

        # If all expected calls were made and current reward is high, return 1.0
        if call_count >= len(golden_answers) and instance_data["reward"] > 0.9:
            return 1.0

        # If some calls were made, return proportional reward
        if call_count > 0:
            completion_ratio = min(call_count, len(golden_answers)) / max(len(golden_answers), 1)
            return instance_data["reward"] * completion_ratio

        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the tool instance resources.

        Args:
            instance_id: The instance id
        """
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
