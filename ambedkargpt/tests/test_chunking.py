import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.chunking import SemanticChunker, BufferMerger


class TestBufferMerger:
    def test_init(self):
        merger = BufferMerger(buffer_size=3)
        assert merger.buffer_size == 3
        
        merger = BufferMerger(buffer_size=0)
        assert merger.buffer_size == 0
    
    def test_merge_empty(self):
        merger = BufferMerger(buffer_size=2)
        result = merger.merge([])
        assert result == []
    
    def test_merge_no_buffer(self):
        merger = BufferMerger(buffer_size=0)
        sentences = ["Sentence one.", "Sentence two.", "Sentence three."]
        result = merger.merge(sentences)
        assert result == sentences
    
    def test_merge_with_buffer(self):
        merger = BufferMerger(buffer_size=1)
        sentences = ["A.", "B.", "C.", "D."]
        result = merger.merge(sentences)
        assert len(result) == 4
        assert "A." in result[0] and "B." in result[0]
        assert "A." in result[1] and "B." in result[1] and "C." in result[1]
    
    def test_merge_for_embedding(self):
        merger = BufferMerger(buffer_size=1)
        sentences = ["First.", "Second.", "Third."]
        result = merger.merge_for_embedding(sentences)
        
        assert len(result) == 3
        assert all('text' in r for r in result)
        assert all('center_idx' in r for r in result)
        assert all('original_sentence' in r for r in result)


class TestSemanticChunker:
    @pytest.fixture
    def chunker(self):
        return SemanticChunker(
            embedding_model="all-MiniLM-L6-v2",
            similarity_threshold=0.5,
            buffer_size=2,
            max_tokens=512,
            overlap_tokens=64
        )
    
    def test_init(self, chunker):
        assert chunker.similarity_threshold == 0.5
        assert chunker.buffer_size == 2
        assert chunker.max_tokens == 512
    
    def test_split_into_sentences(self, chunker):
        text = "This is sentence one. This is sentence two. And here is sentence three."
        sentences = chunker.split_into_sentences(text)
        
        assert len(sentences) == 3
        assert "sentence one" in sentences[0]
    
    def test_get_embeddings(self, chunker):
        texts = ["Hello world.", "This is a test."]
        embeddings = chunker.get_embeddings(texts)
        
        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] > 0  # Has embedding dimensions
    
    def test_calculate_cosine_distances(self, chunker):
        embeddings = chunker.get_embeddings([
            "The cat sat on the mat.",
            "The cat is sitting on the mat.",
            "Quantum physics is fascinating."
        ])
        
        distances = chunker.calculate_cosine_distances(embeddings)
        
        assert len(distances) == 2
        assert distances[0] < distances[1]
    
    def test_chunk_document_basic(self, chunker):
        document = """
        This is the first paragraph about topic A. It discusses various aspects of topic A.
        The topic continues with more information.
        
        Now we move to topic B. This is completely different from topic A.
        Topic B has its own characteristics and features.
        """
        
        chunks = chunker.chunk_document(document)
        
        assert len(chunks) > 0
        assert all('id' in c for c in chunks)
        assert all('text' in c for c in chunks)
        assert all('embedding' in c for c in chunks)
    
    def test_estimate_tokens(self, chunker):
        """Test token estimation."""
        text = "This is a test sentence with some words in it."
        tokens = chunker.estimate_tokens(text)
        assert tokens > 0
        assert tokens == len(text) // 4
    
    def test_split_with_overlap(self, chunker):
        large_text = "Word " * 300  # About 1500 chars
        chunk = {'id': 0, 'text': large_text}
        
        chunker.max_tokens = 100
        sub_chunks = chunker.split_with_overlap(chunk)
        
        assert len(sub_chunks) > 1
        assert all('is_subchunk' in c or 'parent_chunk_id' in c for c in sub_chunks[1:])


class TestSemanticChunkerIntegration:
    
    def test_chunk_real_text(self):
        chunker = SemanticChunker(
            similarity_threshold=0.5,
            buffer_size=3,
            max_tokens=512
        )

        text = """
        Many of us, I dare say, have witnessed local, national or international expositions 
        of material objects that make up the sum total of human civilization. But few can 
        entertain the idea of there being such a thing as an exposition of human institutions.
        
        Caste is a social institution that has profound implications. It affects every aspect
        of social life in India. The caste system is unique in its mechanisms and effects.
        
        To understand caste, we must examine its origin and development. The caste system
        did not emerge suddenly but evolved over centuries through various social processes.
        """
        
        chunks = chunker.chunk_document(text)
        
        assert len(chunks) >= 1
        total_text = " ".join(c['text'] for c in chunks)
        assert "caste" in total_text.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

