# models/english_simulator.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class EnglishSimulator:
    """Fine-tune lightweight English model to simulate Arabic morphology"""
    
    def __init__(self, base_model="gpt2-small", custom_vocab_size=10000):
        # Start with lightweight English model
        self.model = AutoModelForCausalLM.from_pretrained(base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        
        # Add special tokens for morphological simulation
        special_tokens = {
            "additional_special_tokens": [
                "[ROOT_]", "[PATTERN_]", "[MORPH_]",  # Arabic simulation
                "[CONSOLIDATE]", "[WORKING_MEM]", "[LONG_TERM]",
                "[SLOW_ATTN]", "[FAST_ATTN]"
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))