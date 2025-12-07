from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class PromptTemplates:
    SYSTEM_PROMPT_QA = """You are an expert assistant specializing in the works and philosophy of Dr. B.R. Ambedkar. 
You provide accurate, well-informed answers based on the provided context from his writings.
Always cite specific passages when possible and acknowledge if information is not available in the context.
Be respectful and scholarly in your responses."""

    SYSTEM_PROMPT_SUMMARIZER = """You are an expert at summarizing complex academic and philosophical texts.
Create clear, concise summaries that capture the main ideas and key relationships between concepts.
Focus on the essential information while maintaining accuracy."""

    # Answer generation template
    ANSWER_TEMPLATE = """Based on the following context from Dr. B.R. Ambedkar's works, please answer the question.

CONTEXT:
{context}

RELEVANT ENTITIES:
{entities}

QUESTION: {question}

INSTRUCTIONS:
1. Answer based primarily on the provided context
2. If the context doesn't contain enough information, say so clearly
3. Reference specific passages or ideas from the context
4. Be accurate and scholarly in your response

ANSWER:"""

    # Answer template with history
    ANSWER_WITH_HISTORY_TEMPLATE = """Based on the following context from Dr. B.R. Ambedkar's works and our conversation history, please answer the question.

CONVERSATION HISTORY:
{history}

CONTEXT:
{context}

RELEVANT ENTITIES:
{entities}

CURRENT QUESTION: {question}

INSTRUCTIONS:
1. Consider the conversation history for context
2. Answer based primarily on the provided context
3. If the context doesn't contain enough information, say so clearly
4. Reference specific passages or ideas from the context
5. Be accurate and scholarly in your response

ANSWER:"""

    COMMUNITY_SUMMARY_TEMPLATE = """Summarize the following entities and their relationships from a knowledge graph into a coherent paragraph.
Focus on the main themes, key connections, and overall significance.

ENTITIES AND RELATIONSHIPS:
{entities_and_relations}

SUMMARY:"""

    ENTITY_EXTRACTION_TEMPLATE = """Identify the key entities (people, organizations, places, concepts, events) 
and their relationships in the following text.

TEXT:
{text}

List the entities and relationships in the following format:
ENTITIES:
- [Entity Name] (Type): Brief description
...

RELATIONSHIPS:
- [Entity1] -> [relationship] -> [Entity2]
..."""

    CHUNK_REFINEMENT_TEMPLATE = """The following text chunk may contain incomplete or fragmented information.
Please identify if this chunk is:
1. Complete and coherent
2. Missing context at the beginning
3. Missing context at the end
4. Contains unrelated information

TEXT:
{text}

ANALYSIS:"""

    FOLLOWUP_TEMPLATE = """Based on the previous answer and the new question, provide a response.

PREVIOUS CONTEXT:
{previous_context}

PREVIOUS ANSWER:
{previous_answer}

NEW QUESTION: {question}

ANSWER:"""

    @classmethod
    def get_qa_prompt(
        cls,
        question: str,
        context: str,
        entities: List[str] = None,
        history: str = None
    ) -> str:
        """
        Generate a question-answering prompt.
        
        Args:
            question: The user's question
            context: Retrieved context passages
            entities: List of relevant entities
            history: Conversation history
            
        Returns:
            Formatted prompt string
        """
        entities_str = "\n".join([f"- {e}" for e in entities]) if entities else "No specific entities identified"
        
        if history:
            return cls.ANSWER_WITH_HISTORY_TEMPLATE.format(
                history=history,
                context=context,
                entities=entities_str,
                question=question
            )
        else:
            return cls.ANSWER_TEMPLATE.format(
                context=context,
                entities=entities_str,
                question=question
            )
    
    @classmethod
    def get_system_prompt(cls, task: str = "qa") -> str:
        if task == "summarize":
            return cls.SYSTEM_PROMPT_SUMMARIZER
        return cls.SYSTEM_PROMPT_QA
    
    @classmethod
    def get_summary_prompt(cls, entities_and_relations: str) -> str:
        return cls.COMMUNITY_SUMMARY_TEMPLATE.format(
            entities_and_relations=entities_and_relations
        )
    
    @classmethod
    def get_entity_extraction_prompt(cls, text: str) -> str:
        return cls.ENTITY_EXTRACTION_TEMPLATE.format(text=text)
    
    @classmethod
    def get_followup_prompt(
        cls,
        question: str,
        previous_context: str,
        previous_answer: str
    ) -> str:
        return cls.FOLLOWUP_TEMPLATE.format(
            question=question,
            previous_context=previous_context,
            previous_answer=previous_answer
        )
    
    @classmethod
    def format_context(cls, chunks: List[Dict], max_length: int = 4000) -> str:
        context_parts = []
        current_length = 0
        
        for i, chunk in enumerate(chunks):
            text = chunk.get('text', '')
            
            if current_length + len(text) > max_length:
                # Truncate if necessary
                remaining = max_length - current_length
                if remaining > 100:
                    text = text[:remaining] + "..."
                else:
                    break
            
            source = chunk.get('sources', ['document'])
            if isinstance(source, list):
                source = source[0]
            
            context_parts.append(f"[Passage {i+1}]\n{text}")
            current_length += len(text) + 20
        
        return "\n\n".join(context_parts)
    
    @classmethod
    def format_entities(cls, entities: List[Dict]) -> List[str]:
        formatted = []
        for entity in entities:
            text = entity.get('text', entity.get('entity', ''))
            etype = entity.get('type', entity.get('label', 'Entity'))
            formatted.append(f"{text} ({etype})")
        return formatted

