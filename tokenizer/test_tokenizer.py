# tokenizer/test_tokenizer.py
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class TokenizeRequest(BaseModel):
    text: str
    tokenization_level: str = "all"  # surface, roots, patterns, all

@app.post("/tokenize")
async def tokenize_arabic(request: TokenizeRequest):
    """API endpoint for testing tokenizer"""
    tokenizer = MinimalArabicTokenizer()
    
    try:
        tokens = tokenizer.tokenize(request.text)
        
        if request.tokenization_level != "all":
            tokens = {request.tokenization_level: tokens[request.tokenization_level]}
            
        # Calculate compression ratio
        original_length = len(request.text)
        token_count = sum(len(v) for v in tokens.values())
        compression_ratio = original_length / token_count if token_count > 0 else 0
        
        return {
            "tokens": tokens,
            "compression_ratio": round(compression_ratio, 2),
            "original_length": original_length,
            "token_counts": {k: len(v) for k, v in tokens.items()}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run with: uvicorn test_tokenizer:app --reload --port 8001