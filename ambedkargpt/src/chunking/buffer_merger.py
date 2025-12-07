from typing import List
import logging

logger = logging.getLogger(__name__)


class BufferMerger:
    
    def __init__(self, buffer_size: int = 5):
        self.buffer_size = buffer_size
        logger.info(f"BufferMerger initialized with buffer_size={buffer_size}")
    
    def merge(self, sentences: List[str]) -> List[str]:
        if not sentences:
            return []
        
        if self.buffer_size == 0:
            return sentences
        
        merged_sentences = []
        n = len(sentences)
        
        for i in range(n):
            start_idx = max(0, i - self.buffer_size)
            end_idx = min(n, i + self.buffer_size + 1)
            buffer_group = sentences[start_idx:end_idx]
            merged_text = " ".join(buffer_group)
            merged_sentences.append(merged_text)
        
        logger.debug(f"Merged {len(sentences)} sentences into {len(merged_sentences)} buffered groups")
        return merged_sentences
    
    def merge_for_embedding(self, sentences: List[str]) -> List[dict]:
        if not sentences:
            return []
        
        result = []
        n = len(sentences)
        
        for i in range(n):
            start_idx = max(0, i - self.buffer_size)
            end_idx = min(n, i + self.buffer_size + 1)
            
            buffer_group = sentences[start_idx:end_idx]
            merged_text = " ".join(buffer_group)
            
            result.append({
                'text': merged_text,
                'center_idx': i,
                'start_idx': start_idx,
                'end_idx': end_idx - 1,
                'original_sentence': sentences[i]
            })
        
        return result
    
    def set_buffer_size(self, buffer_size: int):
        """Update the buffer size."""
        self.buffer_size = buffer_size
        logger.info(f"Buffer size updated to {buffer_size}")

