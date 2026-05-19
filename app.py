from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import json
import uuid
from datetime import datetime
from rag_system import RAGSystem

app = Flask(__name__)
CORS(app)

app.config['KNOWLEDGE_BASE_DIR'] = os.path.join(os.path.dirname(__file__), 'knowledge_base')
app.config['HISTORY_DIR'] = os.path.join(os.path.dirname(__file__), 'history')

os.makedirs(app.config['KNOWLEDGE_BASE_DIR'], exist_ok=True)
os.makedirs(app.config['HISTORY_DIR'], exist_ok=True)

knowledge_base_path = os.path.join(app.config['KNOWLEDGE_BASE_DIR'], 'pest_database.json')
rag_system = RAGSystem(knowledge_base_path)

conversation_history = {}

def save_history(user_id, question, answer):
    history_file = os.path.join(app.config['HISTORY_DIR'], f'{user_id}.json')
    
    record = {
        'timestamp': datetime.now().isoformat(),
        'question': question,
        'answer': answer
    }
    
    if os.path.exists(history_file):
        with open(history_file, 'r', encoding='utf-8') as f:
            history = json.load(f)
    else:
        history = []
    
    history.append(record)
    
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def load_history(user_id):
    history_file = os.path.join(app.config['HISTORY_DIR'], f'{user_id}.json')
    
    if os.path.exists(history_file):
        with open(history_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    question = data.get('question', '')
    user_id = data.get('user_id', str(uuid.uuid4()))
    
    if not question.strip():
        return jsonify({
            'error': '请输入问题',
            'answer': '',
            'sources': []
        })
    
    result = rag_system.generate_answer(question)
    
    save_history(user_id, question, result)
    
    return jsonify({
        'user_id': user_id,
        'question': question,
        'answer': result['answer'],
        'sources': result['sources'],
        'confidence': result['confidence']
    })

@app.route('/api/suggest', methods=['POST'])
def suggest():
    data = request.get_json()
    symptoms = data.get('symptoms', '')
    
    if not symptoms.strip():
        return jsonify({'error': '请描述症状'})
    
    suggestions = rag_system.suggest_pests(symptoms)
    
    return jsonify({
        'symptoms': symptoms,
        'suggestions': suggestions
    })

@app.route('/api/history', methods=['GET'])
def get_history():
    user_id = request.args.get('user_id')
    
    if not user_id:
        return jsonify({'error': '缺少user_id'})
    
    history = load_history(user_id)
    return jsonify({'history': history})

@app.route('/api/crops', methods=['GET'])
def get_crops():
    crops = []
    for crop_type, crop_data in rag_system.knowledge_base.items():
        crops.append({
            'type': crop_type,
            'name': crop_data['name'],
            'pest_count': len(crop_data.get('pests', []))
        })
    return jsonify({'crops': crops})

@app.route('/api/pests', methods=['GET'])
def get_pests():
    crop_type = request.args.get('crop')
    
    if crop_type and crop_type in rag_system.knowledge_base:
        pests = []
        for pest in rag_system.knowledge_base[crop_type].get('pests', []):
            pests.append({
                'name': pest['name'],
                'alias': pest.get('alias', []),
                'host': pest.get('host', '')
            })
        return jsonify({'pests': pests})
    
    all_pests = []
    for crop_type, crop_data in rag_system.knowledge_base.items():
        for pest in crop_data.get('pests', []):
            all_pests.append({
                'name': pest['name'],
                'alias': pest.get('alias', []),
                'host': pest.get('host', ''),
                'crop': crop_data['name']
            })
    return jsonify({'pests': all_pests})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)