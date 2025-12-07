from typing import List, Dict, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)


class LocalGraphRAGSearch:
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        entity_threshold: float = 0.3,
        document_threshold: float = 0.3,
        top_k: int = 5
    ):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.entity_threshold = entity_threshold  # τ_e
        self.document_threshold = document_threshold  # τ_d
        self.top_k = top_k
        
        logger.info(f"LocalGraphRAGSearch initialized: τ_e={entity_threshold}, "
                   f"τ_d={document_threshold}, top_k={top_k}")
    
    def search(
        self,
        query: str,
        graph_builder,
        history: Optional[str] = None
    ) -> Dict:
        logger.info(f"Performing local search for query: {query[:50]}...")
        combined_query = self._combine_query_history(query, history)
        query_embedding = self.embedding_model.encode(combined_query)
        relevant_entities = self._find_relevant_entities(
            query_embedding, 
            graph_builder
        )
        relevant_chunks = self._find_relevant_chunks(
            relevant_entities,
            graph_builder
        )
        ranked_results = self._rank_results(
            query_embedding,
            relevant_entities,
            relevant_chunks,
            graph_builder
        )
        top_entities = ranked_results['entities'][:self.top_k]
        top_chunks = ranked_results['chunks'][:self.top_k]
        
        logger.info(f"Local search found {len(top_entities)} entities and {len(top_chunks)} chunks")
        
        return {
            'entities': top_entities,
            'chunks': top_chunks,
            'query': query,
            'combined_query': combined_query
        }
    
    def _combine_query_history(self, query: str, history: Optional[str]) -> str:
        if history:
            return f"{history}\n\nCurrent Query: {query}"
        return query
    
    def _find_relevant_entities(
        self, 
        query_embedding: np.ndarray,
        graph_builder
    ) -> List[Dict]:
        relevant_entities = []
        
        for entity_id, entity_embedding in graph_builder.entity_embeddings.items():
            similarity = self._cosine_similarity(query_embedding, entity_embedding)
            if similarity > self.entity_threshold:
                node_data = graph_builder.graph.nodes.get(entity_id, {})
                relevant_entities.append({
                    'entity_id': entity_id,
                    'text': node_data.get('text', entity_id),
                    'type': node_data.get('type', 'UNKNOWN'),
                    'similarity': float(similarity),
                    'chunk_ids': list(graph_builder.entity_to_chunks.get(entity_id, [])),
                    'descriptions': node_data.get('descriptions', [])
                })
        relevant_entities.sort(key=lambda x: x['similarity'], reverse=True)
        
        logger.debug(f"Found {len(relevant_entities)} entities above threshold τ_e={self.entity_threshold}")
        return relevant_entities
    
    def _find_relevant_chunks(
        self,
        relevant_entities: List[Dict],
        graph_builder
    ) -> List[Dict]:
        relevant_chunks = {}
        
        for entity in relevant_entities:
            entity_id = entity['entity_id']
            entity_embedding = graph_builder.entity_embeddings.get(entity_id)
            
            if entity_embedding is None:
                continue
            chunk_ids = entity.get('chunk_ids', [])
            
            for chunk_id in chunk_ids:
                if chunk_id in relevant_chunks:
                    continue
                chunk_embedding = graph_builder.chunk_embeddings.get(chunk_id)
                if chunk_embedding is None:
                    continue
                similarity = self._cosine_similarity(entity_embedding, chunk_embedding)
                if similarity > self.document_threshold:
                    chunk = next(
                        (c for c in graph_builder.chunks if c['id'] == chunk_id),
                        None
                    )
                    
                    if chunk:
                        relevant_chunks[chunk_id] = {
                            'chunk_id': chunk_id,
                            'text': chunk.get('text', ''),
                            'similarity': float(similarity),
                            'related_entity': entity['text'],
                            'related_entity_id': entity_id
                        }
        
        return list(relevant_chunks.values())
    
    def _rank_results(
        self,
        query_embedding: np.ndarray,
        entities: List[Dict],
        chunks: List[Dict],
        graph_builder
    ) -> Dict:
        for chunk in chunks:
            chunk_id = chunk['chunk_id']
            chunk_embedding = graph_builder.chunk_embeddings.get(chunk_id)
            
            if chunk_embedding is not None:
                query_sim = self._cosine_similarity(query_embedding, chunk_embedding)
                chunk['final_score'] = (chunk['similarity'] + query_sim) / 2
            else:
                chunk['final_score'] = chunk['similarity']
        chunks.sort(key=lambda x: x.get('final_score', x['similarity']), reverse=True)

        for entity in entities:
            entity['final_score'] = entity['similarity']
        
        return {
            'entities': entities,
            'chunks': chunks
        }
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        vec1 = np.array(vec1).flatten()
        vec2 = np.array(vec2).flatten()
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    def set_thresholds(self, entity_threshold: float = None, document_threshold: float = None):
        if entity_threshold is not None:
            self.entity_threshold = entity_threshold
            logger.info(f"Entity threshold updated to {entity_threshold}")
        
        if document_threshold is not None:
            self.document_threshold = document_threshold
            logger.info(f"Document threshold updated to {document_threshold}")
    
    def set_top_k(self, top_k: int):
        self.top_k = top_k
        logger.info(f"Top-K updated to {top_k}")

