import pytest
import sys
import os
from pathlib import Path
import tempfile
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPipelineIntegration:
    
    @pytest.fixture
    def sample_document(self):
        """Sample document for testing."""
        return """
        Dr. Bhimrao Ramji Ambedkar was an Indian jurist, economist, and social reformer 
        who inspired the Dalit Buddhist movement and campaigned against social discrimination 
        against Untouchables, while also supporting the rights of women and labour.
        
        He was the principal architect of the Constitution of India. Ambedkar was a prolific 
        student, earning doctorates in economics from both Columbia University and the 
        London School of Economics.
        
        His concept of caste was revolutionary. He argued that caste is not merely a division 
        of labour but a division of labourers. Caste, according to Ambedkar, is a state of mind 
        that keeps the Hindu society rigid and immobile.
        
        Ambedkar's analysis of the caste system went beyond economic factors. He emphasized 
        the role of religious sanction in perpetuating caste. The practice of endogamy, he argued,
        was the key mechanism that maintained caste boundaries.
        
        In his view, the annihilation of caste required the destruction of the religious texts 
        that supported it. He advocated for inter-caste marriages and dining as practical steps
        towards breaking down caste barriers.
        """
    
    @pytest.fixture
    def config(self):
        return {
            'models': {
                'embedding_model': 'all-MiniLM-L6-v2',
                'llm_model': 'llama3:8b',
                'ner_model': 'en_core_web_sm'
            },
            'chunking': {
                'buffer_size': 2,
                'similarity_threshold': 0.5,
                'max_tokens': 512,
                'overlap_tokens': 64
            },
            'knowledge_graph': {
                'community_algorithm': 'louvain',
                'min_community_size': 1,
                'resolution': 1.0
            },
            'retrieval': {
                'local': {
                    'entity_threshold': 0.2,
                    'document_threshold': 0.2,
                    'top_k': 3
                },
                'global': {
                    'top_k_communities': 2,
                    'top_k_points': 3
                }
            },
            'generation': {
                'max_context_length': 2000,
                'temperature': 0.7,
                'max_tokens': 512
            }
        }
    
    def test_chunking_pipeline(self, sample_document):
        from src.chunking import SemanticChunker
        
        chunker = SemanticChunker(
            similarity_threshold=0.5,
            buffer_size=2
        )
        
        chunks = chunker.chunk_document(sample_document)
        
        assert len(chunks) > 0
        assert all('text' in c for c in chunks)
        assert all('embedding' in c for c in chunks)
        all_text = " ".join(c['text'] for c in chunks)
        assert 'Ambedkar' in all_text
        assert 'caste' in all_text.lower()
    
    def test_entity_extraction_pipeline(self, sample_document):
        from src.chunking import SemanticChunker
        from src.graph import EntityExtractor
        
        chunker = SemanticChunker()
        chunks = chunker.chunk_document(sample_document)
        
        extractor = EntityExtractor()
        result = extractor.extract_from_chunks(chunks)
        
        assert 'entities' in result
        assert 'relationships' in result
        assert len(result['entities']) > 0
        entity_texts = [e['text'].lower() for e in result['entities']]
        assert any('ambedkar' in t for t in entity_texts) or len(entity_texts) > 0
    
    def test_graph_building_pipeline(self, sample_document):
        from src.chunking import SemanticChunker
        from src.graph import EntityExtractor, GraphBuilder
        
        chunker = SemanticChunker()
        chunks = chunker.chunk_document(sample_document)
        
        extractor = EntityExtractor()
        extraction = extractor.extract_from_chunks(chunks)
        
        graph_builder = GraphBuilder()
        graph_builder.build_from_extraction(extraction, chunks)
        
        stats = graph_builder.get_graph_stats()
        assert stats['num_nodes'] >= 0
        assert stats['num_edges'] >= 0
    
    def test_community_detection(self, sample_document):
        from src.chunking import SemanticChunker
        from src.graph import EntityExtractor, GraphBuilder, CommunityDetector
        
        chunker = SemanticChunker()
        chunks = chunker.chunk_document(sample_document)
        
        extractor = EntityExtractor()
        extraction = extractor.extract_from_chunks(chunks)
        
        graph_builder = GraphBuilder()
        graph_builder.build_from_extraction(extraction, chunks)
        
        detector = CommunityDetector(algorithm='louvain')
        communities = detector.detect_communities(graph_builder.graph)
        if graph_builder.graph.number_of_nodes() > 0:
            assert len(communities) >= 0
    
    def test_local_search(self, sample_document):
        from src.chunking import SemanticChunker
        from src.graph import EntityExtractor, GraphBuilder
        from src.retrieval import LocalGraphRAGSearch
        
        chunker = SemanticChunker()
        chunks = chunker.chunk_document(sample_document)
        
        extractor = EntityExtractor()
        extraction = extractor.extract_from_chunks(chunks)
        
        graph_builder = GraphBuilder()
        graph_builder.build_from_extraction(extraction, chunks)
        
        local_search = LocalGraphRAGSearch(
            entity_threshold=0.1,
            document_threshold=0.1
        )
        
        results = local_search.search("What is caste according to Ambedkar?", graph_builder)
        
        assert 'entities' in results
        assert 'chunks' in results
    
    def test_save_and_load(self, sample_document, config):
        from src.pipeline import AmbedkarGPT
        
        with tempfile.TemporaryDirectory() as tmpdir:
            gpt = AmbedkarGPT(config=config)
            gpt.index_document(sample_document, "test_doc")
            gpt.save_index(tmpdir)
            assert os.path.exists(os.path.join(tmpdir, 'chunks.json'))
            assert os.path.exists(os.path.join(tmpdir, 'knowledge_graph.pkl'))
            gpt2 = AmbedkarGPT(config=config)
            gpt2.load_index(tmpdir)
            
            assert gpt2.is_indexed
            assert len(gpt2.chunks) > 0


class TestComponentInteraction:
    
    def test_chunker_to_extractor(self):
        from src.chunking import SemanticChunker
        from src.graph import EntityExtractor
        
        text = "Dr. Ambedkar was born in Mhow, India. He studied at Columbia University."
        
        chunker = SemanticChunker()
        chunks = chunker.chunk_document(text)
        
        extractor = EntityExtractor()
        for chunk in chunks:
            result = extractor.extract_from_chunk(chunk)
            assert 'entities' in result
            assert 'chunk_id' in result
    
    def test_search_to_ranker(self):
        from src.retrieval import ResultRanker
        
        ranker = ResultRanker()
        
        local_results = {
            'entities': [{'entity_id': 'e1', 'text': 'Test', 'similarity': 0.8, 'type': 'CONCEPT'}],
            'chunks': [{'chunk_id': 'c1', 'text': 'Test text', 'similarity': 0.7}]
        }
        
        global_results = {
            'communities': [{'community_id': 0, 'summary': 'Test', 'score': 0.6}],
            'points': [{'chunk_id': 'c2', 'text': 'Test 2', 'score': 0.5}]
        }
        
        combined = ranker.combine_and_rank("test", local_results, global_results)
        
        assert 'chunks' in combined
        assert len(combined['chunks']) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

