from typing import List, Dict, Optional, Tuple
import logging

from .llm_client import LLMClient
from .prompt_templates import PromptTemplates

logger = logging.getLogger(__name__)


class AnswerGenerator:
    def __init__(
        self,
        llm_client: LLMClient = None,
        model: str = "llama3:8b",
        max_context_length: int = 4000
    ):
        self.llm_client = llm_client or LLMClient(model=model)
        self.max_context_length = max_context_length
        self.conversation_history = []
        
        logger.info(f"AnswerGenerator initialized with max_context_length={max_context_length}")
    
    def generate_answer(
        self,
        query: str,
        local_results: Dict = None,
        global_results: Dict = None,
        ranked_results: Dict = None,
        include_citations: bool = True
    ) -> Dict:
        logger.info(f"Generating answer for: {query[:50]}...")
        context, entities, sources = self._prepare_context(
            local_results,
            global_results,
            ranked_results
        )
        entity_strings = PromptTemplates.format_entities(entities)
        prompt = PromptTemplates.get_qa_prompt(
            question=query,
            context=context,
            entities=entity_strings,
            history=self._get_recent_history()
        )
        
        system_prompt = PromptTemplates.get_system_prompt("qa")
        try:
            answer = self.llm_client.generate(
                prompt=prompt,
                system_prompt=system_prompt
            )
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            answer = "I apologize, but I encountered an error generating the response. Please try again."
        citations = []
        if include_citations and sources:
            citations = self._generate_citations(sources)
        
        self._update_history(query, answer)
        
        result = {
            'answer': answer,
            'query': query,
            'citations': citations,
            'entities_used': entity_strings,
            'num_sources': len(sources)
        }
        
        logger.info(f"Answer generated with {len(citations)} citations")
        return result
    
    def _prepare_context(
        self,
        local_results: Dict = None,
        global_results: Dict = None,
        ranked_results: Dict = None
    ) -> Tuple[str, List[Dict], List[Dict]]:
        chunks = []
        entities = []
        sources = []
    
        if ranked_results:
            chunks = ranked_results.get('chunks', [])
            entities = ranked_results.get('entities', [])
            sources = chunks
        else:
            if local_results:
                chunks.extend(local_results.get('chunks', []))
                entities.extend(local_results.get('entities', []))
            
            if global_results:
                for point in global_results.get('points', []):
                    chunks.append({
                        'text': point.get('text', ''),
                        'chunk_id': point.get('chunk_id'),
                        'sources': ['global']
                    })
                for comm in global_results.get('communities', [])[:2]:
                    summary = comm.get('summary', '')
                    if summary:
                        chunks.append({
                            'text': f"[Topic Summary] {summary}",
                            'chunk_id': f"community_{comm.get('community_id')}",
                            'sources': ['community']
                        })
            
            sources = chunks
        context = PromptTemplates.format_context(chunks, self.max_context_length)
        
        return context, entities, sources
    
    def _generate_citations(self, sources: List[Dict]) -> List[Dict]:
        citations = []
        
        for i, source in enumerate(sources[:5]):  
            chunk_id = source.get('chunk_id', f'source_{i}')
            text = source.get('text', '')
            excerpt = text[:200] + "..." if len(text) > 200 else text
            
            citations.append({
                'id': i + 1,
                'chunk_id': chunk_id,
                'excerpt': excerpt,
                'source_type': source.get('sources', ['document'])[0] if isinstance(source.get('sources'), list) else 'document'
            })
        
        return citations
    
    def _get_recent_history(self, max_turns: int = 3) -> Optional[str]:
        if not self.conversation_history:
            return None
        
        recent = self.conversation_history[-max_turns:]
        history_parts = []
        
        for turn in recent:
            history_parts.append(f"Q: {turn['query']}")
            history_parts.append(f"A: {turn['answer'][:500]}...")
        
        return "\n".join(history_parts)
    
    def _update_history(self, query: str, answer: str):
        self.conversation_history.append({
            'query': query,
            'answer': answer
        })
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
    
    def generate_with_streaming(
        self,
        query: str,
        local_results: Dict = None,
        global_results: Dict = None,
        ranked_results: Dict = None
    ):
        context, entities, sources = self._prepare_context(
            local_results,
            global_results,
            ranked_results
        )
        
        entity_strings = PromptTemplates.format_entities(entities)
        prompt = PromptTemplates.get_qa_prompt(
            question=query,
            context=context,
            entities=entity_strings,
            history=self._get_recent_history()
        )
        
        system_prompt = PromptTemplates.get_system_prompt("qa")
        
        full_answer = ""
        for chunk in self.llm_client.generate_stream(prompt, system_prompt):
            full_answer += chunk
            yield chunk
        
        self._update_history(query, full_answer)
    
    def clear_history(self):
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def format_answer_with_citations(self, result: Dict) -> str:
        answer = result['answer']
        citations = result.get('citations', [])
        
        if not citations:
            return answer
        citation_text = "\n\n---\n**Sources:**\n"
        for cit in citations:
            citation_text += f"\n[{cit['id']}] {cit['excerpt']}\n"
        
        return answer + citation_text
    
    def generate_summary(self, text: str, max_words: int = 150) -> str:
        return self.llm_client.summarize(text, max_words)

