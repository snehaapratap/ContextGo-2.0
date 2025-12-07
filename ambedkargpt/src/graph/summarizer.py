from typing import List, Dict, Optional
import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)


class CommunitySummarizer:
    
    def __init__(self, llm_client=None, embedding_model: str = "all-MiniLM-L6-v2"):
        self.llm_client = llm_client
        self.embedding_model = SentenceTransformer(embedding_model)
        self.community_summaries = {}
        self.community_embeddings = {}
        
        logger.info("CommunitySummarizer initialized")
    
    def set_llm_client(self, llm_client):
        self.llm_client = llm_client
    
    def _create_community_context(self, graph: nx.Graph, community_nodes: set) -> str:
        context_parts = []
        for node in community_nodes:
            node_data = graph.nodes.get(node, {})
            entity_text = node_data.get('text', node)
            entity_type = node_data.get('type', 'Entity')
            descriptions = node_data.get('descriptions', [])
            
            entity_summary = f"- {entity_text} ({entity_type})"
            if descriptions:
                best_desc = max(descriptions, key=len) if descriptions else ""
                if best_desc:
                    entity_summary += f": {best_desc}"
            
            context_parts.append(entity_summary)
        subgraph = graph.subgraph(community_nodes)
        for u, v, data in subgraph.edges(data=True):
            relations = data.get('relations', ['related to'])
            relation = relations[0] if relations else 'related to'
            
            u_text = graph.nodes[u].get('text', u)
            v_text = graph.nodes[v].get('text', v)
            
            rel_summary = f"- {u_text} {relation} {v_text}"
            context_parts.append(rel_summary)
        
        return "\n".join(context_parts)
    
    def _generate_summary_with_llm(self, context: str, community_id: int) -> str:
        if self.llm_client is None:
            lines = context.split('\n')
            return f"Community {community_id} contains {len(lines)} entities and relationships. " + " ".join(lines[:5])
        
        prompt = f"""Summarize the following entities and relationships from a knowledge graph community into a coherent paragraph that captures the main themes and connections.

Entities and Relationships:
{context}

Summary:"""
        
        try:
            summary = self.llm_client.generate(prompt, max_tokens=300)
            return summary.strip()
        except Exception as e:
            logger.error(f"LLM summarization failed: {e}")
            lines = context.split('\n')
            return f"Community about: {', '.join(lines[:3])}"
    
    def summarize_community(
        self, 
        graph: nx.Graph, 
        community_id: int, 
        community_nodes: set
    ) -> Dict:
        context = self._create_community_context(graph, community_nodes)
        summary = self._generate_summary_with_llm(context, community_id)
        summary_embedding = self.embedding_model.encode(summary)
        entities = [
            {
                'text': graph.nodes[node].get('text', node),
                'type': graph.nodes[node].get('type', 'UNKNOWN')
            }
            for node in community_nodes
        ]
        
        result = {
            'community_id': community_id,
            'summary': summary,
            'embedding': summary_embedding.tolist(),
            'entities': entities,
            'num_entities': len(community_nodes),
            'context': context
        }
        self.community_summaries[community_id] = result
        self.community_embeddings[community_id] = summary_embedding
        
        return result
    
    def summarize_all_communities(
        self, 
        graph: nx.Graph, 
        communities: Dict[int, set],
        min_size: int = 2
    ) -> List[Dict]:
        logger.info(f"Summarizing {len(communities)} communities")
        
        summaries = []
        for comm_id, nodes in communities.items():
            if len(nodes) >= min_size:
                summary = self.summarize_community(graph, comm_id, nodes)
                summaries.append(summary)
                logger.debug(f"Summarized community {comm_id} with {len(nodes)} entities")
        
        logger.info(f"Generated {len(summaries)} community summaries")
        return summaries
    
    def find_similar_communities(self, query: str, top_k: int = 3) -> List[Dict]:
        if not self.community_embeddings:
            return []
        query_embedding = self.embedding_model.encode(query)
        similarities = []
        for comm_id, embedding in self.community_embeddings.items():
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            similarities.append({
                **self.community_summaries[comm_id],
                'similarity': float(similarity)
            })
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similarities[:top_k]
    
    def get_community_summary(self, community_id: int) -> Optional[Dict]:
        return self.community_summaries.get(community_id)
    
    def get_all_summaries(self) -> List[Dict]:
        return list(self.community_summaries.values())

