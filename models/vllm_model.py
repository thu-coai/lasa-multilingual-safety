"""
VLLMModel wrapper for vLLM-based text generation.
"""
from typing import List, Dict, Any, Optional, Union
from vllm import LLM, SamplingParams


class VLLMModel:
    """Wrapper class for vLLM model with chat template support."""
    
    def __init__(
        self,
        model: LLM,
        tokenizer,
        model_name: str,
        generation_config: Optional[SamplingParams] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.eos_token_ids = [self.tokenizer.eos_token_id]
        self.pad_token_id = self.tokenizer.pad_token_id
        
        if generation_config is None:
            self.generation_config = SamplingParams()
        elif isinstance(generation_config, SamplingParams):
            self.generation_config = generation_config
        else:
            self.generation_config = SamplingParams(**generation_config)

    def apply_chat_template(
        self,
        messages: Union[str, List[Dict[str, str]]],
        enable_thinking: bool = False
    ) -> str:
        """
        Apply chat template to messages.
        
        Args:
            messages: Either a string (treated as user message) or list of message dicts
            enable_thinking: Whether to enable thinking mode for Qwen3 models
        
        Returns:
            Formatted prompt string
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        
        # Check if assistant prefill is present
        prefill = messages[-1]['role'] == 'assistant' if messages else False
        
        # Build template kwargs
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": not prefill
        }
        
        # Add enable_thinking for Qwen3 models
        if "qwen3" in self.model_name.lower():
            template_kwargs["enable_thinking"] = enable_thinking
        
        prompt = self.tokenizer.apply_chat_template(messages, **template_kwargs)
        
        # Handle BOS token
        if self.tokenizer.bos_token:
            # Remove duplicate BOS if present
            if prompt.startswith(self.tokenizer.bos_token * 2):
                prompt = prompt[len(self.tokenizer.bos_token):]
            # Add BOS if missing
            elif not prompt.startswith(self.tokenizer.bos_token):
                prompt = self.tokenizer.bos_token + prompt
        
        # Handle EOS token for prefill
        if prefill and self.tokenizer.eos_token:
            if prompt.strip().endswith(self.tokenizer.eos_token):
                idx = prompt.rindex(self.tokenizer.eos_token)
                prompt = prompt[:idx].rstrip()
        
        return prompt
    
    def chat(
        self,
        messages: Union[str, List[Dict[str, str]]],
        enable_thinking: bool = False,
        **kwargs
    ) -> str:
        """
        Generate response for a single conversation.
        
        Args:
            messages: User message string or list of message dicts
            enable_thinking: Whether to enable thinking mode
            **kwargs: Additional generation parameters
        
        Returns:
            Generated text string
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        
        input_text = self.apply_chat_template(messages, enable_thinking=enable_thinking)
        
        # Handle generation config
        if "sampling_params" in kwargs:
            gen_config = kwargs["sampling_params"]
        else:
            gen_config = self.generation_config.clone()
            for k, v in kwargs.items():
                if hasattr(gen_config, k):
                    setattr(gen_config, k, v)
        
        outputs = self.model.generate([input_text], gen_config)
        return outputs[0].outputs[0].text
    
    def batch_chat(
        self,
        batch_messages: List[Union[str, List[Dict[str, str]]]],
        enable_thinking: bool = False,
        **kwargs
    ) -> List[str]:
        """
        Generate responses for multiple conversations.
        
        Args:
            batch_messages: List of user messages or message dicts
            enable_thinking: Whether to enable thinking mode
            **kwargs: Additional generation parameters
        
        Returns:
            List of generated text strings
        """
        input_texts = []
        for messages in batch_messages:
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            
            try:
                input_text = self.apply_chat_template(messages, enable_thinking=enable_thinking)
            except Exception:
                # Fallback: use raw content
                input_text = messages[-1]["content"] if isinstance(messages, list) else messages
            
            input_texts.append(input_text)
        
        # Handle generation config
        if "sampling_params" in kwargs:
            gen_config = kwargs["sampling_params"]
        else:
            gen_config = self.generation_config.clone()
            for k, v in kwargs.items():
                if hasattr(gen_config, k):
                    setattr(gen_config, k, v)
        
        outputs = self.model.generate(input_texts, gen_config)
        return [output.outputs[0].text for output in outputs]

