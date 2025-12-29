#!/usr/bin/env python3
"""
OpenRouter 聊天客户端 - Web 版
支持 Markdown 渲染、代码高亮、流式输出
"""

from flask import Flask, render_template, request, Response, jsonify, stream_with_context
import requests
import json
import os
import sys

app = Flask(__name__)

# ============================================
# 👇 从环境变量获取 API Key 👇
# ============================================
API_KEY = os.getenv('OPENROUTER_API_KEY')

if not API_KEY:
    print("❌ 错误：未设置 OPENROUTER_API_KEY 环境变量")
    print("请先设置环境变量：")
    print("  export OPENROUTER_API_KEY='your-api-key-here'")
    sys.exit(1)

MODELS = {
    "claude-4-sonnet": "anthropic/claude-sonnet-4",
    "claude-4-opus": "anthropic/claude-opus-4",
    "claude-3.5-sonnet": "anthropic/claude-3.5-sonnet",
    "gpt-4o": "openai/gpt-4o",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "gemini-1.5-pro": "google/gemini-pro-1.5",
    "gemini-1.5-flash": "google/gemini-flash-1.5",
    "deepseek-chat": "deepseek/deepseek-chat",
    "deepseek-r1": "deepseek/deepseek-r1",
    "llama-3.1-70b": "meta-llama/llama-3.1-70b-instruct",
    "qwen-2.5-72b": "qwen/qwen-2.5-72b-instruct",
}

# 对话历史存储
conversations = {}


@app.route('/')
def index():
    return render_template('index.html', models=MODELS)


@app.route('/api/models')
def get_models():
    return jsonify(MODELS)


@app.route('/api/chat', methods=['POST'])
def chat():
    """非流式聊天"""
    data = request.json
    message = data.get('message', '')
    model_key = data.get('model', 'claude-4-sonnet')
    session_id = data.get('session_id', 'default')
    
    model = MODELS.get(model_key, model_key)
    
    # 获取或创建对话历史
    if session_id not in conversations:
        conversations[session_id] = []
    
    history = conversations[session_id]
    history.append({"role": "user", "content": message})
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": history,
        "max_tokens": 4096,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        
        result = response.json()
        assistant_message = result['choices'][0]['message']['content']
        
        history.append({"role": "assistant", "content": assistant_message})
        
        return jsonify({
            "success": True,
            "message": assistant_message,
            "model": model
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/chat/stream', methods=['POST'])
def chat_stream():
    """流式聊天"""
    data = request.json
    message = data.get('message', '')
    model_key = data.get('model', 'claude-4-sonnet')
    session_id = data.get('session_id', 'default')
    
    model = MODELS.get(model_key, model_key)
    
    if session_id not in conversations:
        conversations[session_id] = []
    
    history = conversations[session_id]
    history.append({"role": "user", "content": message})
    
    def generate():
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": history,
            "max_tokens": 4096,
            "temperature": 0.7,
            "stream": True
        }
        
        full_response = ""
        
        try:
            with requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                stream=True,
                timeout=120
            ) as response:
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            data_str = line[6:]
                            if data_str == '[DONE]':
                                break
                            try:
                                chunk = json.loads(data_str)
                                if 'choices' in chunk and len(chunk['choices']) > 0:
                                    delta = chunk['choices'][0].get('delta', {})
                                    content = delta.get('content', '')
                                    if content:
                                        full_response += content
                                        yield f"data: {json.dumps({'content': content})}\n\n"
                            except json.JSONDecodeError:
                                continue
                
                # 保存完整回复到历史
                history.append({"role": "assistant", "content": full_response})
                yield f"data: {json.dumps({'done': True})}\n\n"
                
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )


@app.route('/api/clear', methods=['POST'])
def clear_history():
    """清除对话历史"""
    data = request.json
    session_id = data.get('session_id', 'default')
    
    if session_id in conversations:
        conversations[session_id] = []
    
    return jsonify({"success": True})


if __name__ == '__main__':
    print("🚀 启动聊天服务器...")
    print(f"🔑 API Key: {API_KEY[:20]}...")
    print("📍 访问 http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)