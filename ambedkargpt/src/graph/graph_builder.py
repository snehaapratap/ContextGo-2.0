from typing import List, Dict, Optional, Set
import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
import pickle

logger = logging.getLogger(__name__)


class GraphBuilder:
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.graph = nx.Graph()
        self.embedding_model = SentenceTransformer(embedding_model)
        self.entity_embeddings = {}
        self.chunk_to_entities = {}
        self.entity_to_chunks = {}
        
        logger.info("GraphBuilder initialized")
    
    def add_entity(
        self, 
        entity_text: str, 
        entity_type: str,
        descriptions: List[str] = None,
        chunk_ids: List[str] = None
    ) -> str:
        node_id = entity_text.lower().strip()
        
        if node_id in self.graph:
            node_data = self.graph.nodes[node_id]
            if descriptions:
                existing_descs = node_data.get('descriptions', [])
                node_data['descriptions'] = list(set(existing_descs + descriptions))
            if chunk_ids:
                existing_chunks = node_data.get('chunk_ids', [])
                node_data['chunk_ids'] = list(set(existing_chunks + chunk_ids))
        else:
            self.graph.add_node(
                node_id,
                text=entity_text,
                type=entity_type,
                descriptions=descriptions or [],
                chunk_ids=chunk_ids or []
            )
            embedding = self.embedding_model.encode(entity_text)
            self.entity_embeddings[node_id] = embedding
            self.graph.nodes[node_id]['embedding'] = embedding.tolist()
        if chunk_ids:
            for chunk_id in chunk_ids:
                if chunk_id not in self.chunk_to_entities:
                    self.chunk_to_entities[chunk_id] = set()
                self.chunk_to_entities[chunk_id].add(node_id)
            
            self.entity_to_chunks[node_id] = set(chunk_ids)
        
        return node_id
    
    def add_relationship(
        self,
        source: str,
        target: str,
        relation_type: str,
        description: str = "",
        weight: float = 1.0
    ):
        source_id = source.lower().strip()
        target_id = target.lower().strip()
        if source_id in self.graph and target_id in self.graph:
            if self.graph.has_edge(source_id, target_id):
                edge_data = self.graph[source_id][target_id]
                edge_data['weight'] = edge_data.get('weight', 1.0) + weight
                if description and description not in edge_data.get('descriptions', []):
                    edge_data.setdefault('descriptions', []).append(description)
                if relation_type not in edge_data.get('relations', []):
                    edge_data.setdefault('relations', []).append(relation_type)
            else:
                self.graph.add_edge(
                    source_id,
                    target_id,
                    relations=[relation_type],
                    descriptions=[description] if description else [],
                    weight=weight
                )
    
    def build_from_extraction(self, extraction_result: Dict, chunks: List[Dict]):
        logger.info("Building knowledge graph from extraction results")
        chunk_lookup = {c['id']: c for c in chunks}
        for entity in extraction_result['entities']:
            self.add_entity(
                entity_text=entity['text'],
                entity_type=entity['label'],
                descriptions=entity.get('descriptions', []),
                chunk_ids=entity.get('chunk_ids', [])
            )
        for rel in extraction_result['relationships']:
            self.add_relationship(
                source=rel['source'],
                target=rel['target'],
                relation_type=rel['relation'],
                description=rel.get('description', '')
            )
        self.chunks = chunks
        self.chunk_embeddings = {}
        for chunk in chunks:
            if 'embedding' in chunk:
                self.chunk_embeddings[chunk['id']] = np.array(chunk['embedding'])
        
        logger.info(f"Built graph with {self.graph.number_of_nodes()} nodes "
                   f"and {self.graph.number_of_edges()} edges")
    
    def get_entity_neighbors(self, entity: str, depth: int = 1) -> Set[str]:
        entity_id = entity.lower().strip()
        if entity_id not in self.graph:
            return set()
        
        neighbors = set()
        current_level = {entity_id}
        
        for _ in range(depth):
            next_level = set()
            for node in current_level:
                for neighbor in self.graph.neighbors(node):
                    if neighbor not in neighbors and neighbor != entity_id:
                        neighbors.add(neighbor)
                        next_level.add(neighbor)
            current_level = next_level
        
        return neighbors
    
    def get_entity_subgraph(self, entities: List[str]) -> nx.Graph:
        entity_ids = [e.lower().strip() for e in entities if e.lower().strip() in self.graph]
        return self.graph.subgraph(entity_ids)
    
    def get_entity_similarity(self, entity1: str, entity2: str) -> float:
        id1 = entity1.lower().strip()
        id2 = entity2.lower().strip()
        
        if id1 not in self.entity_embeddings or id2 not in self.entity_embeddings:
            return 0.0
        
        emb1 = self.entity_embeddings[id1]
        emb2 = self.entity_embeddings[id2]
        
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)
    
    def find_similar_entities(self, query: str, top_k: int = 5) -> List[Dict]:
        query_embedding = self.embedding_model.encode(query)
        
        similarities = []
        for entity_id, embedding in self.entity_embeddings.items():
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            similarities.append({
                'entity_id': entity_id,
                'entity': self.graph.nodes[entity_id].get('text', entity_id),
                'type': self.graph.nodes[entity_id].get('type', 'UNKNOWN'),
                'similarity': float(similarity)
            })
        
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]
    
    def get_graph_stats(self) -> Dict:
        return {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph) if self.graph.number_of_nodes() > 0 else 0,
            'num_connected_components': nx.number_connected_components(self.graph) if self.graph.number_of_nodes() > 0 else 0,
            'entity_types': self._count_entity_types()
        }
    
    def _count_entity_types(self) -> Dict[str, int]:
        type_counts = {}
        for node_id in self.graph.nodes():
            entity_type = self.graph.nodes[node_id].get('type', 'UNKNOWN')
            type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
        return type_counts
    
    def save(self, filepath: str):
        data = {
            'graph': self.graph,
            'entity_embeddings': self.entity_embeddings,
            'chunk_to_entities': self.chunk_to_entities,
            'entity_to_chunks': self.entity_to_chunks,
            'chunks': getattr(self, 'chunks', []),
            'chunk_embeddings': getattr(self, 'chunk_embeddings', {})
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Graph saved to {filepath}")
    
    def load(self, filepath: str):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.graph = data['graph']
        self.entity_embeddings = data['entity_embeddings']
        self.chunk_to_entities = data['chunk_to_entities']
        self.entity_to_chunks = data['entity_to_chunks']
        self.chunks = data.get('chunks', [])
        self.chunk_embeddings = data.get('chunk_embeddings', {})
        
        logger.info(f"Graph loaded from {filepath}")

