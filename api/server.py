# api/server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from typing import Optional

app = FastAPI(title="Neuro-Mimetic LLM API")

# Load model (simplified version)
model = None  # Will be loaded at startup
tokenizer = None

class GenerationRequest(BaseModel):
    prompt: str
    max_length: int = 100
    temperature: float = 0.7
    use_slow_attention: bool = True
    consolidation_steps: int = 3

class TrainingRequest(BaseModel):
    text: str
    learning_rate: float = 1e-5
    consolidation: bool = True

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global model, tokenizer
    # Load minimal model for testing
    model = MinimalNeuroMimeticLLM()
    # In production, load trained weights
    model.load_state_dict(torch.load("checkpoints/minimal.pth"))
    model.eval()

@app.post("/generate")
async def generate_text(request: GenerationRequest):
    """Generate text with neuro-mimetic model"""
    try:
        # Tokenize
        inputs = tokenizer(request.prompt, return_tensors="pt")
        
        # Generate with slow attention if requested
        with torch.no_grad():
            if request.use_slow_attention:
                # Multi-step generation with consolidation
                output = generate_with_consolidation(
                    model, inputs, 
                    max_length=request.max_length,
                    temperature=request.temperature,
                    consolidation_steps=request.consolidation_steps
                )
            else:
                # Standard generation
                output = model.generate(
                    **inputs,
                    max_length=request.max_length,
                    temperature=request.temperature
                )
        
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        return {
            "generated_text": text,
            "attention_patterns": get_attention_patterns(model),
            "context_used": len(output[0])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train_online")
async def online_training(request: TrainingRequest):
    """Online learning endpoint (simulating lifelong learning)"""
    try:
        # Tokenize training data
        inputs = tokenizer(request.text, return_tensors="pt")
        
        # Single training step (simulating daily learning)
        optimizer = torch.optim.Adam(model.parameters(), lr=request.learning_rate)
        loss = train_single_step(model, inputs, optimizer, request.consolidation)
        
        return {
            "loss": loss,
            "context_size_increased": model.context_size,
            "plasticity_rate": model.attention_plasticity.mean().item()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model_status")
async def get_status():
    """Get current model state"""
    return {
        "context_window": model.context_size if hasattr(model, 'context_size') else 1024,
        "total_tokens_processed": model.total_tokens if hasattr(model, 'total_tokens') else 0,
        "memory_usage": get_memory_usage(model),
        "attention_plasticity": model.attention_plasticity.tolist() if hasattr(model, 'attention_plasticity') else []
    }