# tokenizer/arabic_morphological.py
import regex as re
from typing import List, Dict

class MinimalArabicTokenizer:
    """Lightweight Arabic morphological tokenizer for prototyping"""
    
    def __init__(self):
        # Core Arabic morphological patterns (simplified)
        self.patterns = {
            "roots": self.extract_roots,
            "verbal_patterns": self.identify_verbal_patterns,
            "noun_patterns": self.identify_noun_patterns
        }
        
        # Simple root dictionary (can be expanded)
        self.root_dict = self.load_common_roots()
        
    def tokenize(self, text: str) -> Dict[str, List[str]]:
        """Multi-scale tokenization"""
        tokens = {
            "surface": self.surface_tokenize(text),      # BPE-like
            "roots": self.extract_roots(text),           # Semantic cores
            "patterns": self.extract_patterns(text),     # Morph templates
            "positions": self.get_positional_encoding(text)  # For attention
        }
        return tokens
    
    def surface_tokenize(self, text: str) -> List[str]:
        """Simple surface tokenization (placeholder for BPE)"""
        # Split by whitespace and punctuation
        words = re.findall(r'\b\w+\b|[^\w\s]', text)
        return words
    
    def extract_roots(self, text: str) -> List[str]:
        """Extract triliteral roots (3-letter semantic cores)"""
        roots = []
        words = text.split()
        
        for word in words:
            # Simple root extraction (heuristic-based)
            root = self.guess_root(word)
            if root:
                roots.append(f"[ROOT_{root}]")
                
        return roots
    
    def guess_root(self, word: str) -> str:
        """Heuristic root extraction for prototyping"""
        # Remove common prefixes and suffixes
        cleaned = re.sub(r'^(ال|و|ف|ب|ك|ل|س|ي)', '', word)
        cleaned = re.sub(r'(ون|ين|ات|ة|ي|و|ا)$', '', cleaned)
        
        # Try to find 3-letter root
        if len(cleaned) >= 3:
            # Take middle 3 characters (heuristic)
            start = (len(cleaned) - 3) // 2
            return cleaned[start:start+3]
        return cleaned