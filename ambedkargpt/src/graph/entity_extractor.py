from typing import List, Dict, Set, Tuple, Optional
import spacy
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class EntityExtractor:
    RELEVANT_ENTITY_TYPES = {
        'PERSON', 'ORG', 'GPE', 'LOC', 'EVENT', 'WORK_OF_ART',
        'LAW', 'NORP', 'FAC', 'PRODUCT', 'DATE', 'TIME'
    }
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        logger.info(f"Loading spaCy model: {model_name}")
        
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            logger.warning(f"Model {model_name} not found. Downloading...")
            spacy.cli.download(model_name)
            self.nlp = spacy.load(model_name)
        
        logger.info(f"EntityExtractor initialized with model {model_name}")
    
    def extract_entities(self, text: str) -> List[Dict]:
        doc = self.nlp(text)
        
        entities = []
        seen_entities = set()
        
        for ent in doc.ents:
            if ent.label_ in self.RELEVANT_ENTITY_TYPES:
                entity_text = ent.text.strip()
                entity_key = (entity_text.lower(), ent.label_)
                
                if entity_key not in seen_entities:
                    seen_entities.add(entity_key)
                    entities.append({
                        'text': entity_text,
                        'label': ent.label_,
                        'start': ent.start_char,
                        'end': ent.end_char,
                        'description': self._get_entity_description(ent, doc)
                    })
        
        logger.debug(f"Extracted {len(entities)} entities from text")
        return entities
    
    def _get_entity_description(self, entity, doc) -> str:
        for sent in doc.sents:
            if entity.start >= sent.start and entity.end <= sent.end:
                return sent.text
        return ""
    
    def extract_relationships(self, text: str) -> List[Dict]:
        doc = self.nlp(text)
        relationships = []
        for token in doc:
            if token.pos_ == 'VERB':
                subjects = [child for child in token.children if child.dep_ in ('nsubj', 'nsubjpass')]
                objects = [child for child in token.children if child.dep_ in ('dobj', 'pobj', 'attr')]
                
                for subj in subjects:
                    for obj in objects:
                        subj_ent = self._get_entity_for_token(subj, doc)
                        obj_ent = self._get_entity_for_token(obj, doc)
                        
                        if subj_ent and obj_ent:
                            relationships.append({
                                'source': subj_ent,
                                'target': obj_ent,
                                'relation': token.lemma_,
                                'description': token.sent.text
                            })
        
        entities = self.extract_entities(text)
        for sent in doc.sents:
            sent_entities = [e for e in entities if e['start'] >= sent.start_char and e['end'] <= sent.end_char]
            for i, ent1 in enumerate(sent_entities):
                for ent2 in sent_entities[i+1:]:
                    existing = any(
                        r['source'] == ent1['text'] and r['target'] == ent2['text']
                        for r in relationships
                    )
                    if not existing:
                        relationships.append({
                            'source': ent1['text'],
                            'target': ent2['text'],
                            'relation': 'related_to',
                            'description': sent.text,
                            'source_type': ent1['label'],
                            'target_type': ent2['label']
                        })
        
        logger.debug(f"Extracted {len(relationships)} relationships from text")
        return relationships
    
    def _get_entity_for_token(self, token, doc) -> Optional[str]:
        for ent in doc.ents:
            if token.i >= ent.start and token.i < ent.end:
                return ent.text
        return token.text if token.pos_ in ('PROPN', 'NOUN') else None
    
    def extract_from_chunk(self, chunk: Dict) -> Dict:
        text = chunk['text']
        chunk_id = chunk.get('id', 'unknown')
        
        entities = self.extract_entities(text)
        relationships = self.extract_relationships(text)
        for entity in entities:
            entity['chunk_id'] = chunk_id
        
        for rel in relationships:
            rel['chunk_id'] = chunk_id
        
        return {
            'chunk_id': chunk_id,
            'entities': entities,
            'relationships': relationships
        }
    
    def extract_from_chunks(self, chunks: List[Dict]) -> Dict:
        all_entities = []
        all_relationships = []
        entity_to_chunks = defaultdict(set)
        
        for chunk in chunks:
            result = self.extract_from_chunk(chunk)
            
            for entity in result['entities']:
                all_entities.append(entity)
                entity_to_chunks[entity['text'].lower()].add(chunk['id'])
            
            all_relationships.extend(result['relationships'])
        unique_entities = {}
        for entity in all_entities:
            key = entity['text'].lower()
            if key not in unique_entities:
                unique_entities[key] = {
                    'text': entity['text'],
                    'label': entity['label'],
                    'chunk_ids': list(entity_to_chunks[key]),
                    'descriptions': [entity['description']]
                }
            else:
                if entity['description'] not in unique_entities[key]['descriptions']:
                    unique_entities[key]['descriptions'].append(entity['description'])
        unique_relationships = {}
        for rel in all_relationships:
            key = (rel['source'].lower(), rel['target'].lower(), rel['relation'])
            if key not in unique_relationships:
                unique_relationships[key] = rel
        
        logger.info(f"Extracted {len(unique_entities)} unique entities and "
                   f"{len(unique_relationships)} unique relationships from {len(chunks)} chunks")
        
        return {
            'entities': list(unique_entities.values()),
            'relationships': list(unique_relationships.values()),
            'entity_to_chunks': dict(entity_to_chunks)
        }
    
    def extract_key_concepts(self, text: str, top_n: int = 10) -> List[str]:
        doc = self.nlp(text)
        noun_chunks = [chunk.text.strip() for chunk in doc.noun_chunks]
        freq = defaultdict(int)
        for chunk in noun_chunks:
            freq[chunk.lower()] += 1
        sorted_concepts = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [concept for concept, _ in sorted_concepts[:top_n]]

