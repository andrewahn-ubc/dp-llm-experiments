"""Local causal-LM chat target for PyRIT (supports LoRA + fixed CUDA device).

HuggingFaceChatTarget keeps a class-level model cache and calls ``.to("cuda")``,
which breaks dual-GPU attacker/target setups. This target loads one model onto
an explicit device and optionally merges a PEFT adapter.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pyrit.exceptions import EmptyResponseException, pyrit_target_retry
from pyrit.models import ComponentIdentifier, Message, construct_response_from_request
from pyrit.prompt_target.common.prompt_target import PromptTarget
from pyrit.prompt_target.common.target_capabilities import TargetCapabilities
from pyrit.prompt_target.common.target_configuration import TargetConfiguration
from pyrit.prompt_target.common.utils import limit_requests_per_minute

logger = logging.getLogger(__name__)


class LocalCausalLMChatTarget(PromptTarget):
    _DEFAULT_CONFIGURATION: TargetConfiguration = TargetConfiguration(
        capabilities=TargetCapabilities(
            supports_multi_turn=True,
            supports_editable_history=True,
            supports_system_prompt=True,
        )
    )

    def __init__(
        self,
        *,
        model_path: str,
        adapter_path: str | None = None,
        device: str = "cuda:0",
        torch_dtype: Any | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        trust_remote_code: bool = True,
        attn_implementation: str | None = None,
        system_prompt: str | None = None,
        max_requests_per_minute: int | None = None,
    ) -> None:
        super().__init__(
            max_requests_per_minute=max_requests_per_minute,
            model_name=model_path,
            custom_configuration=self._DEFAULT_CONFIGURATION,
        )
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.device = device
        self.torch_dtype = torch_dtype or torch.bfloat16
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.trust_remote_code = trust_remote_code
        self.attn_implementation = attn_implementation
        self.fixed_system_prompt = system_prompt

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        load_kwargs: dict[str, Any] = {
            "torch_dtype": self.torch_dtype,
            "trust_remote_code": trust_remote_code,
            "low_cpu_mem_usage": True,
        }
        if attn_implementation:
            load_kwargs["attn_implementation"] = attn_implementation

        logger.info("Loading model %s on %s (adapter=%s)", model_path, device, adapter_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        if adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)
        self.model = model.to(device)
        self.model.eval()

    def _build_identifier(self) -> ComponentIdentifier:
        return self._create_identifier(
            params={
                "model_path": self.model_path,
                "adapter_path": self.adapter_path,
                "device": self.device,
                "max_new_tokens": self.max_new_tokens,
            }
        )

    def _build_chat_messages(self, *, normalized_conversation: list[Message]) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if self.fixed_system_prompt:
            messages.append({"role": "system", "content": self.fixed_system_prompt})
        for msg in normalized_conversation:
            piece = msg.message_pieces[0]
            role = piece.api_role
            # Skip system from history if we already injected a fixed empty/protocol prompt.
            if role == "system" and self.fixed_system_prompt is not None:
                continue
            content = piece.converted_value or ""
            messages.append({"role": role, "content": content})
        return messages

    @limit_requests_per_minute
    @pyrit_target_retry
    async def _send_prompt_to_target_async(
        self, *, normalized_conversation: list[Message]
    ) -> list[Message]:
        request = normalized_conversation[-1].message_pieces[0]
        messages = self._build_chat_messages(normalized_conversation=normalized_conversation)

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.do_sample:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = self.top_p
        else:
            gen_kwargs["temperature"] = None
            gen_kwargs["top_p"] = None

        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        text = self.tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()
        if not text:
            raise EmptyResponseException

        response = construct_response_from_request(
            request=request,
            response_text_pieces=[text],
            prompt_metadata={
                "model_path": self.model_path,
                "adapter_path": self.adapter_path or "",
                "device": self.device,
                "effective_generation_config": json.dumps(
                    {k: str(v) for k, v in gen_kwargs.items()}
                ),
            },
        )
        return [response]
