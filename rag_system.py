import json
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class RAGSystem:
    def __init__(self, knowledge_base_path):
        self.knowledge_base = self._load_knowledge_base(knowledge_base_path)
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
        self.document_vectors = None
        self.document_texts = []
        self.document_metadata = []
        self._build_index()
    
    def _load_knowledge_base(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _build_index(self):
        for crop_type, crop_data in self.knowledge_base.items():
            for pest in crop_data.get('pests', []):
                text = f"{pest['name']} {' '.join(pest.get('alias', []))} {pest.get('symptoms', '')} {pest.get('host', '')} {pest.get('classification', '')}"
                text = re.sub(r'[^\w\s\u4e00-\u9fa5]', '', text)
                self.document_texts.append(text)
                self.document_metadata.append({
                    'crop': crop_data['name'],
                    'pest': pest
                })
        
        if self.document_texts:
            self.document_vectors = self.vectorizer.fit_transform(self.document_texts)
    
    def _preprocess_query(self, query):
        return re.sub(r'[^\w\s\u4e00-\u9fa5]', '', query)
    
    def retrieve(self, query, top_k=3):
        if self.document_vectors is None or not self.document_texts:
            return []
        
        query_processed = self._preprocess_query(query)
        query_vector = self.vectorizer.transform([query_processed])
        
        similarities = cosine_similarity(query_vector, self.document_vectors).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:
                results.append({
                    'similarity': float(similarities[idx]),
                    'metadata': self.document_metadata[idx]
                })
        
        return results
    
    def generate_answer(self, query):
        retrieved = self.retrieve(query)
        
        if not retrieved:
            return {
                'answer': '抱歉，我暂时无法回答这个问题。请尝试描述更具体的症状或病虫害名称。',
                'sources': [],
                'confidence': 0
            }
        
        top_result = retrieved[0]
        pest = top_result['metadata']['pest']
        crop = top_result['metadata']['crop']
        
        answer_parts = []
        
        if 'symptoms' in query.lower() or '症状' in query or '表现' in query:
            answer_parts.append(f"**{pest['name']}**（别名：{', '.join(pest.get('alias', []))}）")
            answer_parts.append(f"**危害症状**：{pest.get('symptoms', '')}")
            answer_parts.append(f"**危害作物**：{pest.get('host', '')}")
        elif '防治' in query or '怎么治' in query or '用药' in query or '防治方法' in query:
            answer_parts.append(f"针对**{pest['name']}**的防治方法如下：")
            if pest.get('agricultural_control'):
                answer_parts.append(f"**农业防治**：{pest['agricultural_control']}")
            if pest.get('chemical_control'):
                answer_parts.append(f"**化学防治**：{pest['chemical_control']}")
            if pest.get('biological_control'):
                answer_parts.append(f"**生物防治**：{pest['biological_control']}")
            if pest.get('prevention'):
                answer_parts.append(f"**预防建议**：{pest['prevention']}")
        elif '发生' in query or '规律' in query or '什么时候' in query or '条件' in query:
            answer_parts.append(f"**{pest['name']}**的发生规律：")
            if pest.get('occurrence'):
                answer_parts.append(f"**发生条件**：{pest['occurrence']}")
            if pest.get('transmission'):
                answer_parts.append(f"**传播途径**：{pest['transmission']}")
        else:
            answer_parts.append(f"**{pest['name']}**（{pest.get('classification', '')}）")
            answer_parts.append(f"**危害作物**：{pest.get('host', '')}")
            answer_parts.append(f"**危害症状**：{pest.get('symptoms', '')}")
            if pest.get('occurrence'):
                answer_parts.append(f"**发生规律**：{pest.get('occurrence')}")
            answer_parts.append(f"**防治方法**：")
            if pest.get('agricultural_control'):
                answer_parts.append(f"- 农业防治：{pest['agricultural_control']}")
            if pest.get('chemical_control'):
                answer_parts.append(f"- 化学防治：{pest['chemical_control']}")
            if pest.get('biological_control'):
                answer_parts.append(f"- 生物防治：{pest['biological_control']}")
        
        answer = '\n\n'.join(answer_parts)
        
        sources = [{
            'crop': crop,
            'pest_name': pest['name'],
            'similarity': top_result['similarity']
        }]
        
        return {
            'answer': answer,
            'sources': sources,
            'confidence': min(top_result['similarity'], 1.0)
        }
    
    def suggest_pests(self, symptoms):
        retrieved = self.retrieve(symptoms, top_k=5)
        
        suggestions = []
        for result in retrieved:
            if result['similarity'] > 0.05:
                pest = result['metadata']['pest']
                suggestions.append({
                    'name': pest['name'],
                    'alias': pest.get('alias', []),
                    'crop': result['metadata']['crop'],
                    'symptoms': pest.get('symptoms', '')[:50] + '...',
                    'similarity': result['similarity']
                })
        
        return suggestions