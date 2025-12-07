import pytest
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval import LocalGraphRAGSearch, GlobalGraphRAGSearch, ResultRanker


class MockGraphBuilder:
    def __init__(self):
        self.entity_embeddings = {
            'ambedkar': np.random.rand(384),
            'caste': np.random.rand(384),
            'india': np.random.rand(384),
        }
        self.entity_to_chunks = {
            'ambedkar': ['chunk_0', 'chunk_1'],
            'caste': ['chunk_0', 'chunk_2'],
            'india': ['chunk_1', 'chunk_2'],
        }
        self.chunk_embeddings = {
            'chunk_0': np.random.rand(384),
            'chunk_1': np.random.rand(384),
            'chunk_2': np.random.rand(384),
        }
        self.chunks = [
            {'id': 'chunk_0', 'text': 'Dr. Ambedkar wrote about caste system.'},
            {'id': 'chunk_1', 'text': 'Ambedkar was born in India.'},
            {'id': 'chunk_2', 'text': 'The caste system in India has deep roots.'},
        ]
        
        # Mock graph
        import networkx as nx
        self.graph = nx.Graph()
        for entity_id, embedding in self.entity_embeddings.items():
            self.graph.add_node(entity_id, text=entity_id.title(), type='CONCEPT', embedding=embedding.tolist())


class MockCommunitySummarizer:
    
    def __init__(self):
        self.community_embeddings = {
            0: np.random.rand(384),
            1: np.random.rand(384),
        }
        self.community_summaries = {
            0: {
                'community_id': 0,
                'summary': 'This community discusses caste and social structure.',
                'entities': [{'text': 'caste'}, {'text': 'ambedkar'}],
                'embedding': np.random.rand(384).tolist()
            },
            1: {
                'community_id': 1,
                'summary': 'This community discusses India and geography.',
                'entities': [{'text': 'india'}],
                'embedding': np.random.rand(384).tolist()
            },
        }
    
    def find_similar_communities(self, query, top_k=3):
        return list(self.community_summaries.values())[:top_k]


class TestLocalGraphRAGSearch:
    
    @pytest.fixture
    def local_search(self):
        return LocalGraphRAGSearch(
            entity_threshold=0.1,  
            document_threshold=0.1,
            top_k=3
        )
    
    @pytest.fixture
    def mock_graph(self):
        return MockGraphBuilder()
    
    def test_init(self, local_search):
        assert local_search.entity_threshold == 0.1
        assert local_search.document_threshold == 0.1
        assert local_search.top_k == 3
    
    def test_search_returns_results(self, local_search, mock_graph):
        results = local_search.search("What is caste?", mock_graph)
        
        assert 'entities' in results
        assert 'chunks' in results
        assert 'query' in results
    
    def test_search_with_history(self, local_search, mock_graph):
        results = local_search.search(
            "Tell me more",
            mock_graph,
            history="Previous discussion about Ambedkar"
        )
        
        assert 'combined_query' in results
        assert 'Previous discussion' in results['combined_query']
    
    def test_set_thresholds(self, local_search):
        local_search.set_thresholds(entity_threshold=0.5, document_threshold=0.4)
        assert local_search.entity_threshold == 0.5
        assert local_search.document_threshold == 0.4
    
    def test_cosine_similarity(self, local_search):
        vec1 = np.array([1, 0, 0])
        vec2 = np.array([1, 0, 0])
        sim = local_search._cosine_similarity(vec1, vec2)
        assert abs(sim - 1.0) < 1e-6
        
        vec3 = np.array([0, 1, 0])
        sim2 = local_search._cosine_similarity(vec1, vec3)
        assert abs(sim2) < 1e-6


class TestGlobalGraphRAGSearch:
    @pytest.fixture
    def global_search(self):
        return GlobalGraphRAGSearch(
            top_k_communities=2,
            top_k_points=3
        )
    
    @pytest.fixture
    def mock_summarizer(self):
        return MockCommunitySummarizer()
    
    @pytest.fixture
    def mock_graph(self):
        return MockGraphBuilder()
    
    def test_init(self, global_search):
        assert global_search.top_k_communities == 2
        assert global_search.top_k_points == 3
    
    def test_search_returns_communities(self, global_search, mock_summarizer, mock_graph):
        results = global_search.search(
            "What about caste?",
            mock_summarizer,
            mock_graph
        )
        
        assert 'communities' in results
        assert 'points' in results
        assert 'query' in results
    
    def test_set_parameters(self, global_search):
        global_search.set_parameters(top_k_communities=5, top_k_points=10)
        assert global_search.top_k_communities == 5
        assert global_search.top_k_points == 10

class TestResultRanker:    
    @pytest.fixture
    def ranker(self):
        return ResultRanker(local_weight=0.6, global_weight=0.4)
    
    def test_init(self, ranker):
        assert ranker.local_weight == 0.6
        assert ranker.global_weight == 0.4
    
    def test_combine_and_rank(self, ranker):
        local_results = {
            'entities': [
                {'entity_id': 'test', 'text': 'Test', 'type': 'CONCEPT', 'similarity': 0.8}
            ],
            'chunks': [
                {'chunk_id': 'c1', 'text': 'Test chunk', 'similarity': 0.7}
            ]
        }
        global_results = {
            'communities': [
                {'community_id': 0, 'summary': 'Test summary', 'score': 0.6}
            ],
            'points': [
                {'chunk_id': 'c2', 'text': 'Another chunk', 'score': 0.5}
            ]
        }
        
        results = ranker.combine_and_rank(
            "test query",
            local_results,
            global_results,
            top_k=5
        )
        
        assert 'chunks' in results
        assert 'entities' in results
        assert 'communities' in results
    
    def test_rerank(self, ranker):
        chunks = [
            {'chunk_id': 'c1', 'text': 'About caste system'},
            {'chunk_id': 'c2', 'text': 'Weather forecast today'},
        ]
        
        reranked = ranker.rerank("What is caste?", chunks, top_k=2)
        
        assert len(reranked) == 2
        assert all('rerank_score' in c for c in reranked)
    
    def test_format_context_for_llm(self, ranker):
        results = {
            'entities': [{'text': 'Ambedkar', 'type': 'PERSON'}],
            'chunks': [{'text': 'Sample text about caste.', 'chunk_id': 'c1'}],
            'communities': [{'summary': 'Caste discussion'}]
        }
        
        context = ranker.format_context_for_llm(results, max_context_length=1000)
        
        assert 'Ambedkar' in context
        assert 'Sample text' in context
    
    def test_set_weights(self, ranker):
        ranker.set_weights(local_weight=0.8, global_weight=0.2)
        assert ranker.local_weight == 0.8
        assert ranker.global_weight == 0.2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

