from typing import List, Dict, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import re
import nltk
from nltk.tokenize import sent_tokenize

from .buffer_merger import BufferMerger

logger = logging.getLogger(__name__)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


class SemanticChunker:
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.5,
        buffer_size: int = 5,
        max_tokens: int = 1024,
        overlap_tokens: int = 128
    ):
        logger.info(f"Initializing SemanticChunker with model={embedding_model}")
        
        self.embedding_model = SentenceTransformer(embedding_model)
        self.similarity_threshold = similarity_threshold
        self.buffer_size = buffer_size
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        
        self.buffer_merger = BufferMerger(buffer_size)
        
        logger.info(f"SemanticChunker initialized: threshold={similarity_threshold}, "
                   f"buffer_size={buffer_size}, max_tokens={max_tokens}")
    
    def split_into_sentences(self, document: str) -> List[str]:
        document = self._clean_text(document)
        sentences = sent_tokenize(document)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        logger.debug(f"Split document into {len(sentences)} sentences")
        return sentences
    
    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?;:\'"()-]', '', text)
        return text.strip()
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
        return np.array(embeddings)
    
    def calculate_cosine_distances(self, embeddings: np.ndarray) -> List[float]:
        distances = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            distance = 1 - sim
            distances.append(distance)
        
        return distances
    
    def create_chunks(
        self, 
        sentences: List[str], 
        distances: List[float]
    ) -> List[Dict]:
        chunks = []
        current_chunk = {
            'sentences': [sentences[0]],
            'indices': [0]
        }
        
        for i, distance in enumerate(distances):
            if distance < self.similarity_threshold:
                current_chunk['sentences'].append(sentences[i + 1])
                current_chunk['indices'].append(i + 1)
            else:
                chunks.append(current_chunk)
                current_chunk = {
                    'sentences': [sentences[i + 1]],
                    'indices': [i + 1]
                }
        
        if current_chunk['sentences']:
            chunks.append(current_chunk)
        final_chunks = []
        for i, chunk in enumerate(chunks):
            text = " ".join(chunk['sentences'])
            final_chunks.append({
                'id': i,
                'text': text,
                'sentence_indices': chunk['indices'],
                'num_sentences': len(chunk['sentences'])
            })
        
        logger.debug(f"Created {len(final_chunks)} initial chunks")
        return final_chunks
    
    def estimate_tokens(self, text: str) -> int:
        return len(text) // 4
    
    def split_with_overlap(self, chunk: Dict) -> List[Dict]:
        text = chunk['text']
        tokens = self.estimate_tokens(text)
        
        if tokens <= self.max_tokens:
            return [chunk]
        sub_chunks = []
        words = text.split()
        words_per_chunk = (self.max_tokens * 4) // 5  
        overlap_words = (self.overlap_tokens * 4) // 5
        
        start = 0
        sub_idx = 0
        
        while start < len(words):
            end = min(start + words_per_chunk, len(words))
            sub_text = " ".join(words[start:end])
            
            sub_chunks.append({
                'id': f"{chunk['id']}_{sub_idx}",
                'text': sub_text,
                'parent_chunk_id': chunk['id'],
                'is_subchunk': True,
                'subchunk_index': sub_idx
            })
            
            sub_idx += 1
            start = end - overlap_words  
            
            if start >= len(words) - overlap_words:
                break
        
        return sub_chunks
    
    def chunk_document(self, document: str) -> List[Dict]:
        logger.info("Starting semantic chunking pipeline")
        sentences = self.split_into_sentences(document)
        if len(sentences) < 2:
            return [{'id': 0, 'text': document, 'sentence_indices': [0], 'num_sentences': 1}]
        
        logger.info(f"Document split into {len(sentences)} sentences")
        merged_sentences = self.buffer_merger.merge(sentences)
        embeddings = self.get_embeddings(merged_sentences)
        logger.info(f"Generated {len(embeddings)} embeddings")
        distances = self.calculate_cosine_distances(embeddings)
        chunks = self.create_chunks(sentences, distances)  
        final_chunks = []
        for chunk in chunks:
            sub_chunks = self.split_with_overlap(chunk)
            final_chunks.extend(sub_chunks)
        chunk_texts = [c['text'] for c in final_chunks]
        chunk_embeddings = self.get_embeddings(chunk_texts)
        
        for i, chunk in enumerate(final_chunks):
            chunk['embedding'] = chunk_embeddings[i].tolist()
        
        logger.info(f"Semantic chunking complete: {len(final_chunks)} chunks created")
        return final_chunks
    
    def chunk_documents(self, documents: List[str]) -> List[Dict]:
        all_chunks = []
        
        for doc_idx, doc in enumerate(documents):
            chunks = self.chunk_document(doc)
            for chunk in chunks:
                chunk['document_id'] = doc_idx
                chunk['id'] = f"doc{doc_idx}_chunk{chunk['id']}"
            
            all_chunks.extend(chunks)
        
        logger.info(f"Processed {len(documents)} documents into {len(all_chunks)} total chunks")
        return all_chunks
    
    def get_chunk_similarity(self, chunk1: Dict, chunk2: Dict) -> float:
        emb1 = np.array(chunk1['embedding'])
        emb2 = np.array(chunk2['embedding'])
        return cosine_similarity([emb1], [emb2])[0][0]

