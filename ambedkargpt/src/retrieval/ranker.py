from typing import List, Dict, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)


class ResultRanker:
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        local_weight: float = 0.6,
        global_weight: float = 0.4
    ):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.local_weight = local_weight
        self.global_weight = global_weight
        
        logger.info(f"ResultRanker initialized: local_weight={local_weight}, "
                   f"global_weight={global_weight}")
    
    def combine_and_rank(
        self,
        query: str,
        local_results: Dict,
        global_results: Dict,
        top_k: int = 5
    ) -> Dict:
        query_embedding = self.embedding_model.encode(query)
        all_chunks = {}
        for chunk in local_results.get('chunks', []):
            chunk_id = chunk['chunk_id']
            if chunk_id not in all_chunks:
                all_chunks[chunk_id] = {
                    'chunk_id': chunk_id,
                    'text': chunk['text'],
                    'local_score': chunk.get('final_score', chunk.get('similarity', 0)),
                    'global_score': 0,
                    'sources': ['local']
                }
            else:
                all_chunks[chunk_id]['local_score'] = chunk.get('final_score', chunk.get('similarity', 0))
                all_chunks[chunk_id]['sources'].append('local')
        for point in global_results.get('points', []):
            chunk_id = point['chunk_id']
            if chunk_id not in all_chunks:
                all_chunks[chunk_id] = {
                    'chunk_id': chunk_id,
                    'text': point['text'],
                    'local_score': 0,
                    'global_score': point.get('score', 0),
                    'sources': ['global'],
                    'community_id': point.get('community_id')
                }
            else:
                all_chunks[chunk_id]['global_score'] = point.get('score', 0)
                if 'global' not in all_chunks[chunk_id]['sources']:
                    all_chunks[chunk_id]['sources'].append('global')
                all_chunks[chunk_id]['community_id'] = point.get('community_id')
        for chunk_id, chunk in all_chunks.items():
            combined_score = (
                self.local_weight * chunk['local_score'] +
                self.global_weight * chunk['global_score']
            )
            if len(chunk['sources']) > 1:
                combined_score *= 1.2
            
            chunk['combined_score'] = combined_score
        ranked_chunks = sorted(
            all_chunks.values(),
            key=lambda x: x['combined_score'],
            reverse=True
        )[:top_k]
        all_entities = {}
        for entity in local_results.get('entities', []):
            entity_id = entity['entity_id']
            if entity_id not in all_entities:
                all_entities[entity_id] = entity
        
        ranked_entities = sorted(
            all_entities.values(),
            key=lambda x: x.get('final_score', x.get('similarity', 0)),
            reverse=True
        )[:top_k]
        
        return {
            'chunks': ranked_chunks,
            'entities': ranked_entities,
            'communities': global_results.get('communities', []),
            'query': query
        }
    
    def rerank(
        self,
        query: str,
        chunks: List[Dict],
        top_k: int = 5
    ) -> List[Dict]:
        if not chunks:
            return []
        
        query_embedding = self.embedding_model.encode(query)
        
        for chunk in chunks:
            text = chunk.get('text', '')
            if text:
                chunk_embedding = self.embedding_model.encode(text)
                similarity = self._cosine_similarity(query_embedding, chunk_embedding)
                chunk['rerank_score'] = float(similarity)
            else:
                chunk['rerank_score'] = chunk.get('combined_score', 0)
        
        reranked = sorted(chunks, key=lambda x: x['rerank_score'], reverse=True)
        return reranked[:top_k]
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    def format_context_for_llm(
        self,
        ranked_results: Dict,
        max_context_length: int = 4000
    ) -> str:
        context_parts = []
        current_length = 0
        entities = ranked_results.get('entities', [])
        if entities:
            entity_texts = [f"- {e['text']} ({e['type']})" for e in entities[:5]]
            entity_section = "Key Entities:\n" + "\n".join(entity_texts)
            context_parts.append(entity_section)
            current_length += len(entity_section)
        communities = ranked_results.get('communities', [])
        if communities:
            for comm in communities[:2]:
                summary = comm.get('summary', '')
                if summary and current_length + len(summary) < max_context_length:
                    context_parts.append(f"\nTopic Summary:\n{summary}")
                    current_length += len(summary) + 20
        chunks = ranked_results.get('chunks', [])
        context_parts.append("\nRelevant Passages:")
        
        for i, chunk in enumerate(chunks):
            chunk_text = chunk.get('text', '')
            if current_length + len(chunk_text) > max_context_length:
                break
            
            source_info = ", ".join(chunk.get('sources', ['unknown']))
            context_parts.append(f"\n[Passage {i+1} - Source: {source_info}]\n{chunk_text}")
            current_length += len(chunk_text) + 50
        
        return "\n".join(context_parts)
    
    def set_weights(self, local_weight: float = None, global_weight: float = None):
        if local_weight is not None:
            self.local_weight = local_weight
        if global_weight is not None:
            self.global_weight = global_weight
        
        # Normalize
        total = self.local_weight + self.global_weight
        if total > 0:
            self.local_weight /= total
            self.global_weight /= total

