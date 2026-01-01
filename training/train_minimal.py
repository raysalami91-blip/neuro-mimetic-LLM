# training/train_minimal.py
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup

class NeuroMimeticTrainer:
    """Minimal trainer for prototype"""
    
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        
        # Training state
        self.context_size = 1024  # Start small
        self.curriculum_stage = 0
        self.total_tokens_seen = 0
        
    def train_epoch(self, dataloader, optimizer, scheduler):
        self.model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            
            # Forward pass with multi-scale processing
            working_output, longterm_output = self.model(input_ids)
            
            # Language modeling loss
            logits = self.model.lm_head(working_output)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                input_ids.view(-1)
            )
            
            # Add consolidation loss (align working & long-term)
            consolidation_loss = F.mse_loss(working_output, longterm_output)
            total_loss = loss + 0.1 * consolidation_loss
            
            # Backward with plasticity constraints
            total_loss.backward()
            
            # Apply gradient clipping for slow learning
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Update context size gradually
            self.total_tokens_seen += input_ids.size(0) * input_ids.size(1)
            self.update_context_size()
            
        return total_loss.item()
    
    def update_context_size(self):
        """Gradually increase context window"""
        # Human-like growth: fast early, then slow
        growth_rate = max(0.01, 1.0 / (1.0 + self.curriculum_stage * 0.1))
        self.context_size = min(
            int(self.context_size * (1 + growth_rate)),
            16384  # Maximum for prototype
        )
        
        # Update model if needed
        if hasattr(self.model, 'update_context'):
            self.model.update_context(self.context_size)