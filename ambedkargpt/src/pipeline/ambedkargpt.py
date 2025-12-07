from typing import List, Dict, Optional
import os
import json
import yaml
import logging
from pathlib import Path
import pickle
try:
    import pdfplumber
    PDF_LIBRARY = "pdfplumber"
except ImportError:
    try:
        from pypdf import PdfReader
        PDF_LIBRARY = "pypdf"
    except ImportError:
        PDF_LIBRARY = None

from ..chunking import SemanticChunker
from ..graph import EntityExtractor, GraphBuilder, CommunityDetector, CommunitySummarizer
from ..retrieval import LocalGraphRAGSearch, GlobalGraphRAGSearch, ResultRanker
from ..llm import LLMClient, PromptTemplates, AnswerGenerator

logger = logging.getLogger(__name__)


class AmbedkarGPT:
    
    def __init__(self, config_path: str = None, config: Dict = None):
        self.config = self._load_config(config_path, config)
        self._setup_logging()
        
        logger.info("Initializing AmbedkarGPT...")
        self._init_components()

        self.is_indexed = False
        self.chunks = []
        
        logger.info("AmbedkarGPT initialized successfully")
    
    def _load_config(self, config_path: str, config: Dict) -> Dict:
        if config:
            return config
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return {
            'models': {
                'embedding_model': 'all-MiniLM-L6-v2',
                'llm_model': 'llama3:8b',
                'ner_model': 'en_core_web_sm'
            },
            'chunking': {
                'buffer_size': 5,
                'similarity_threshold': 0.5,
                'max_tokens': 1024,
                'overlap_tokens': 128
            },
            'knowledge_graph': {
                'community_algorithm': 'leiden',
                'min_community_size': 2,
                'resolution': 1.0
            },
            'retrieval': {
                'local': {
                    'entity_threshold': 0.3,
                    'document_threshold': 0.3,
                    'top_k': 5
                },
                'global': {
                    'top_k_communities': 3,
                    'top_k_points': 5
                }
            },
            'generation': {
                'max_context_length': 4096,
                'temperature': 0.7,
                'max_tokens': 1024
            },
            'paths': {
                'data_dir': 'data',
                'processed_dir': 'data/processed',
                'chunks_file': 'data/processed/chunks.json',
                'graph_file': 'data/processed/knowledge_graph.pkl'
            }
        }
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get('logging', {})
        level = getattr(logging, log_config.get('level', 'INFO'))
        format_str = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        logging.basicConfig(level=level, format=format_str)
    
    def _init_components(self):
        chunk_config = self.config.get('chunking', {})
        model_config = self.config.get('models', {})
        retrieval_config = self.config.get('retrieval', {})
        gen_config = self.config.get('generation', {})
        kg_config = self.config.get('knowledge_graph', {})
        self.chunker = SemanticChunker(
            embedding_model=model_config.get('embedding_model', 'all-MiniLM-L6-v2'),
            similarity_threshold=chunk_config.get('similarity_threshold', 0.5),
            buffer_size=chunk_config.get('buffer_size', 5),
            max_tokens=chunk_config.get('max_tokens', 1024),
            overlap_tokens=chunk_config.get('overlap_tokens', 128)
        )
        self.entity_extractor = EntityExtractor(
            model_name=model_config.get('ner_model', 'en_core_web_sm')
        )
        self.graph_builder = GraphBuilder(
            embedding_model=model_config.get('embedding_model', 'all-MiniLM-L6-v2')
        )
        self.community_detector = CommunityDetector(
            algorithm=kg_config.get('community_algorithm', 'leiden'),
            resolution=kg_config.get('resolution', 1.0)
        )

        self.llm_client = LLMClient(
            model=model_config.get('llm_model', 'llama3:8b'),
            temperature=gen_config.get('temperature', 0.7),
            max_tokens=gen_config.get('max_tokens', 1024)
        )
        self.summarizer = CommunitySummarizer(
            llm_client=self.llm_client,
            embedding_model=model_config.get('embedding_model', 'all-MiniLM-L6-v2')
        )
        local_config = retrieval_config.get('local', {})
        self.local_search = LocalGraphRAGSearch(
            embedding_model=model_config.get('embedding_model', 'all-MiniLM-L6-v2'),
            entity_threshold=local_config.get('entity_threshold', 0.3),
            document_threshold=local_config.get('document_threshold', 0.3),
            top_k=local_config.get('top_k', 5)
        )
        global_config = retrieval_config.get('global', {})
        self.global_search = GlobalGraphRAGSearch(
            embedding_model=model_config.get('embedding_model', 'all-MiniLM-L6-v2'),
            top_k_communities=global_config.get('top_k_communities', 3),
            top_k_points=global_config.get('top_k_points', 5)
        )
        
        self.ranker = ResultRanker(
            embedding_model=model_config.get('embedding_model', 'all-MiniLM-L6-v2')
        )

        self.answer_generator = AnswerGenerator(
            llm_client=self.llm_client,
            max_context_length=gen_config.get('max_context_length', 4096)
        )
    
    def load_pdf(self, pdf_path: str) -> str:
        logger.info(f"Loading PDF: {pdf_path}")
        
        if PDF_LIBRARY == "pdfplumber":
            text_parts = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
            return "\n\n".join(text_parts)
        
        elif PDF_LIBRARY == "pypdf":
            reader = PdfReader(pdf_path)
            text_parts = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            return "\n\n".join(text_parts)
        
        else:
            raise ImportError("No PDF library available. Install pdfplumber or pypdf.")
    
    def index_document(self, document: str, document_id: str = "doc_0") -> Dict:
        logger.info(f"Indexing document {document_id}...")
        logger.info("Step 1: Semantic chunking...")
        chunks = self.chunker.chunk_document(document)
        

        for chunk in chunks:
            chunk['document_id'] = document_id
            if not isinstance(chunk['id'], str) or not chunk['id'].startswith('doc'):
                chunk['id'] = f"{document_id}_chunk{chunk['id']}"
        
        self.chunks = chunks
        logger.info(f"Created {len(chunks)} semantic chunks")
        
        logger.info("Step 2: Extracting entities and relationships...")
        extraction_result = self.entity_extractor.extract_from_chunks(chunks)
        logger.info(f"Extracted {len(extraction_result['entities'])} entities and "
                   f"{len(extraction_result['relationships'])} relationships")
        
        logger.info("Step 3: Building knowledge graph...")
        self.graph_builder.build_from_extraction(extraction_result, chunks)
        
        logger.info("Step 4: Detecting communities...")
        communities = self.community_detector.detect_communities(self.graph_builder.graph)
        
        min_size = self.config.get('knowledge_graph', {}).get('min_community_size', 2)
        self.community_detector.merge_small_communities(self.graph_builder.graph, min_size)
        
        logger.info(f"Detected {len(self.community_detector.communities)} communities")
        
        logger.info("Step 5: Generating community summaries...")
        summaries = self.summarizer.summarize_all_communities(
            self.graph_builder.graph,
            self.community_detector.communities,
            min_size=min_size
        )
        logger.info(f"Generated {len(summaries)} community summaries")
        
        self.is_indexed = True
        stats = {
            'num_chunks': len(chunks),
            'num_entities': len(extraction_result['entities']),
            'num_relationships': len(extraction_result['relationships']),
            'num_graph_nodes': self.graph_builder.graph.number_of_nodes(),
            'num_graph_edges': self.graph_builder.graph.number_of_edges(),
            'num_communities': len(self.community_detector.communities),
            'num_summaries': len(summaries)
        }
        
        logger.info(f"Indexing complete: {stats}")
        return stats
    
    def index_pdf(self, pdf_path: str) -> Dict:
        document = self.load_pdf(pdf_path)
        document_id = Path(pdf_path).stem
        return self.index_document(document, document_id)
    
    def query(
        self,
        question: str,
        search_type: str = "hybrid",
        history: str = None
    ) -> Dict:

        if not self.is_indexed:
            return {
                'answer': "Please index a document first using index_pdf() or index_document().",
                'error': True
            }
        
        logger.info(f"Processing query: {question[:50]}...")
        local_results = None
        global_results = None
        
        if search_type in ['local', 'hybrid']:
            logger.info("Performing local search (Equation 4)...")
            local_results = self.local_search.search(
                question,
                self.graph_builder,
                history
            )
        
        if search_type in ['global', 'hybrid']:
            logger.info("Performing global search (Equation 5)...")
            global_results = self.global_search.search(
                question,
                self.summarizer,
                self.graph_builder
            )
        if search_type == 'hybrid' and local_results and global_results:
            logger.info("Combining and ranking results...")
            ranked_results = self.ranker.combine_and_rank(
                question,
                local_results,
                global_results
            )
        else:
            ranked_results = None
        logger.info("Generating answer...")
        result = self.answer_generator.generate_answer(
            question,
            local_results=local_results,
            global_results=global_results,
            ranked_results=ranked_results,
            include_citations=True
        )
        result['search_type'] = search_type
        result['local_entities'] = len(local_results.get('entities', [])) if local_results else 0
        result['global_communities'] = len(global_results.get('communities', [])) if global_results else 0
        
        return result
    
    def save_index(self, directory: str = None):
        if directory is None:
            directory = self.config.get('paths', {}).get('processed_dir', 'data/processed')
        
        os.makedirs(directory, exist_ok=True)
        chunks_file = os.path.join(directory, 'chunks.json')
        with open(chunks_file, 'w') as f:
            serializable_chunks = []
            for chunk in self.chunks:
                chunk_copy = chunk.copy()
                if 'embedding' in chunk_copy:
                    chunk_copy['embedding'] = list(chunk_copy['embedding']) if hasattr(chunk_copy['embedding'], 'tolist') else chunk_copy['embedding']
                serializable_chunks.append(chunk_copy)
            json.dump(serializable_chunks, f, indent=2)

        graph_file = os.path.join(directory, 'knowledge_graph.pkl')
        self.graph_builder.save(graph_file)
        
        # Save community data
        community_file = os.path.join(directory, 'communities.pkl')
        with open(community_file, 'wb') as f:
            pickle.dump({
                'communities': self.community_detector.communities,
                'node_to_community': self.community_detector.node_to_community,
                'summaries': self.summarizer.community_summaries,
                'summary_embeddings': {k: v.tolist() if hasattr(v, 'tolist') else v 
                                       for k, v in self.summarizer.community_embeddings.items()}
            }, f)
        
        logger.info(f"Index saved to {directory}")
    
    def load_index(self, directory: str = None):
        if directory is None:
            directory = self.config.get('paths', {}).get('processed_dir', 'data/processed')
        chunks_file = os.path.join(directory, 'chunks.json')
        with open(chunks_file, 'r') as f:
            self.chunks = json.load(f)
        graph_file = os.path.join(directory, 'knowledge_graph.pkl')
        self.graph_builder.load(graph_file)
        community_file = os.path.join(directory, 'communities.pkl')
        with open(community_file, 'rb') as f:
            community_data = pickle.load(f)
        
        self.community_detector.communities = community_data['communities']
        self.community_detector.node_to_community = community_data['node_to_community']
        self.summarizer.community_summaries = community_data['summaries']
        self.summarizer.community_embeddings = community_data['summary_embeddings']
        
        self.is_indexed = True
        logger.info(f"Index loaded from {directory}")
    
    def get_stats(self) -> Dict:
        if not self.is_indexed:
            return {'indexed': False}
        
        return {
            'indexed': True,
            'num_chunks': len(self.chunks),
            'graph_stats': self.graph_builder.get_graph_stats(),
            'community_stats': self.community_detector.get_community_stats()
        }
    
    def clear(self):
        self.chunks = []
        self.graph_builder = GraphBuilder(
            embedding_model=self.config.get('models', {}).get('embedding_model', 'all-MiniLM-L6-v2')
        )
        self.community_detector = CommunityDetector(
            algorithm=self.config.get('knowledge_graph', {}).get('community_algorithm', 'leiden')
        )
        self.summarizer = CommunitySummarizer(
            llm_client=self.llm_client
        )
        self.answer_generator.clear_history()
        self.is_indexed = False
        logger.info("Pipeline cleared")

