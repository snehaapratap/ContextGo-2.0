from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)


class GlobalGraphRAGSearch:
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        top_k_communities: int = 3,
        top_k_points: int = 5
    ):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.top_k_communities = top_k_communities
        self.top_k_points = top_k_points
        
        logger.info(f"GlobalGraphRAGSearch initialized: top_k_communities={top_k_communities}, "
                   f"top_k_points={top_k_points}")
    
    def search(
        self,
        query: str,
        community_summarizer,
        graph_builder
    ) -> Dict:
        logger.info(f"Performing global search for query: {query[:50]}...")
        query_embedding = self.embedding_model.encode(query)
        top_communities = self._find_top_communities(
            query_embedding,
            community_summarizer
        )
        scored_points = self._extract_and_score_points(
            query_embedding,
            top_communities,
            graph_builder
        )
        top_points = sorted(
            scored_points,
            key=lambda x: x['score'],
            reverse=True
        )[:self.top_k_points]
        
        logger.info(f"Global search found {len(top_communities)} communities and {len(top_points)} points")
        
        return {
            'communities': top_communities,
            'points': top_points,
            'query': query
        }
    
    def _find_top_communities(
        self,
        query_embedding: np.ndarray,
        community_summarizer
    ) -> List[Dict]:
        top_communities = community_summarizer.find_similar_communities(
            "",  
            top_k=self.top_k_communities
        )
        
        if not top_communities and community_summarizer.community_embeddings:
            communities = []
            for comm_id, embedding in community_summarizer.community_embeddings.items():
                similarity = self._cosine_similarity(query_embedding, embedding)
                summary_data = community_summarizer.community_summaries.get(comm_id, {})
                communities.append({
                    'community_id': comm_id,
                    'summary': summary_data.get('summary', ''),
                    'entities': summary_data.get('entities', []),
                    'similarity': float(similarity),
                    'score': float(similarity)
                })
            
            communities.sort(key=lambda x: x['similarity'], reverse=True)
            top_communities = communities[:self.top_k_communities]
        
        logger.debug(f"Found {len(top_communities)} top communities")
        return top_communities
    
    def _extract_and_score_points(
        self,
        query_embedding: np.ndarray,
        top_communities: List[Dict],
        graph_builder
    ) -> List[Dict]:
        scored_points = []
        seen_chunks = set()
        
        for community in top_communities:
            community_id = community.get('community_id')
            community_score = community.get('score', community.get('similarity', 0))
            entities = community.get('entities', [])
            
            for entity in entities:
                entity_text = entity.get('text', '') if isinstance(entity, dict) else entity
                entity_id = entity_text.lower().strip()
                chunk_ids = graph_builder.entity_to_chunks.get(entity_id, [])
                
                for chunk_id in chunk_ids:
                    if chunk_id in seen_chunks:
                        continue
                    seen_chunks.add(chunk_id)
                    chunk = next(
                        (c for c in graph_builder.chunks if c['id'] == chunk_id),
                        None
                    )
                    
                    if chunk:
                        chunk_embedding = graph_builder.chunk_embeddings.get(chunk_id)
                        
                        if chunk_embedding is not None:
                            point_score = self._cosine_similarity(
                                query_embedding, 
                                chunk_embedding
                            )
                        else:
                            point_score = community_score * 0.5
                        
                        scored_points.append({
                            'chunk_id': chunk_id,
                            'text': chunk.get('text', ''),
                            'score': float(point_score),
                            'community_id': community_id,
                            'community_score': float(community_score),
                            'related_entity': entity_text
                        })
        
        return scored_points
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        vec1 = np.array(vec1).flatten()
        vec2 = np.array(vec2).flatten()
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    def search_with_direct_embedding(
        self,
        query: str,
        community_summaries: List[Dict],
        chunks: List[Dict],
        chunk_embeddings: Dict[str, np.ndarray]
    ) -> Dict:
        query_embedding = self.embedding_model.encode(query)
        scored_communities = []
        for summary in community_summaries:
            if 'embedding' in summary:
                comm_embedding = np.array(summary['embedding'])
                similarity = self._cosine_similarity(query_embedding, comm_embedding)
                scored_communities.append({
                    **summary,
                    'similarity': float(similarity),
                    'score': float(similarity)
                })
        
        scored_communities.sort(key=lambda x: x['score'], reverse=True)
        top_communities = scored_communities[:self.top_k_communities]
        scored_points = []
        for chunk in chunks:
            chunk_id = chunk.get('id')
            chunk_embedding = chunk_embeddings.get(chunk_id)
            
            if chunk_embedding is not None:
                score = self._cosine_similarity(query_embedding, chunk_embedding)
                scored_points.append({
                    'chunk_id': chunk_id,
                    'text': chunk.get('text', ''),
                    'score': float(score)
                })
        
        scored_points.sort(key=lambda x: x['score'], reverse=True)
        top_points = scored_points[:self.top_k_points]
        
        return {
            'communities': top_communities,
            'points': top_points,
            'query': query
        }
    
    def set_parameters(self, top_k_communities: int = None, top_k_points: int = None):
        if top_k_communities is not None:
            self.top_k_communities = top_k_communities
        if top_k_points is not None:
            self.top_k_points = top_k_points

