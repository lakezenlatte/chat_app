#!/usr/bin/env python3
"""
增强版 AI 助手
支持：联网搜索、文件操作、代码编译运行、Markdown 渲染、图片上传
"""

from flask import Flask, render_template, request, Response, jsonify, stream_with_context
import requests
import json
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime
import threading
import base64

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# ============================================
# 配置
# ============================================
API_KEY = os.getenv('OPENROUTER_API_KEY')

if not API_KEY:
    print("❌ 错误：未设置 OPENROUTER_API_KEY 环境变量")
    print("请先设置环境变量：")
    print("  export OPENROUTER_API_KEY='your-api-key-here'")
    sys.exit(1)

# 默认工作目录
DEFAULT_WORK_DIR = os.path.expanduser('~/work')
os.makedirs(DEFAULT_WORK_DIR, exist_ok=True)

# 用户项目路径存储
user_project_paths = {}

MODELS = {
    "claude-4-sonnet": "anthropic/claude-sonnet-4",
    "claude-4-opus": "anthropic/claude-opus-4",
    "claude-3.5-sonnet": "anthropic/claude-3.5-sonnet",
    "gpt-4o": "openai/gpt-4o",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "deepseek-chat": "deepseek/deepseek-chat",
    "deepseek-r1": "deepseek/deepseek-r1",
}

conversations = {}

# 存储活动的请求，用于取消
active_requests = {}

# ============================================
# 工具函数定义
# ============================================

def get_current_time():
    """获取当前时间"""
    now = datetime.now()
    weekdays = ['星期一', '星期二', '星期三', '星期四', '星期五', '星期六', '星期日']
    weekday = weekdays[now.weekday()]
    
    return f"当前时间: {now.strftime('%Y年%m月%d日')} {weekday} {now.strftime('%H:%M:%S')}"


def web_search(query):
    """网络搜索工具"""
    try:
        url = "https://api.duckduckgo.com/"
        params = {
            'q': query,
            'format': 'json',
            'no_html': 1,
            'skip_disambig': 1
        }
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        results = []
        if data.get('AbstractText'):
            results.append(f"摘要: {data['AbstractText']}")
        
        for topic in data.get('RelatedTopics', [])[:5]:
            if 'Text' in topic:
                results.append(topic['Text'])
        
        return "\n".join(results) if results else "未找到相关结果"
    except Exception as e:
        return f"搜索失败: {str(e)}"


def read_file(filepath, session_id='default'):
    """读取文件内容"""
    try:
        work_dir = user_project_paths.get(session_id, DEFAULT_WORK_DIR)
        full_path = os.path.join(work_dir, filepath)
        
        if not os.path.exists(full_path):
            return f"错误: 文件不存在 {full_path}"
        
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return f"文件内容 ({filepath}):\n```\n{content}\n```"
    except Exception as e:
        return f"读取文件失败: {str(e)}"


def write_file(filepath, content, session_id='default'):
    """写入文件"""
    try:
        work_dir = user_project_paths.get(session_id, DEFAULT_WORK_DIR)
        full_path = os.path.join(work_dir, filepath)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"成功写入文件: {full_path}"
    except Exception as e:
        return f"写入文件失败: {str(e)}"


def list_files(directory=".", session_id='default'):
    """列出目录文件"""
    try:
        work_dir = user_project_paths.get(session_id, DEFAULT_WORK_DIR)
        full_path = os.path.join(work_dir, directory)
        
        if not os.path.exists(full_path):
            return f"错误: 目录不存在 {full_path}"
        
        items = []
        for item in sorted(os.listdir(full_path)):
            item_path = os.path.join(full_path, item)
            if os.path.isdir(item_path):
                items.append(f"📁 {item}/")
            else:
                size = os.path.getsize(item_path)
                items.append(f"📄 {item} ({size} bytes)")
        
        return f"目录 {directory} (项目路径: {work_dir}):\n" + "\n".join(items)
    except Exception as e:
        return f"列出文件失败: {str(e)}"


def execute_command(command, cwd=None, session_id='default'):
    """执行系统命令"""
    try:
        work_dir = user_project_paths.get(session_id, DEFAULT_WORK_DIR)
        work_path = os.path.join(work_dir, cwd) if cwd else work_dir
        
        # 安全检查
        dangerous_commands = ['rm -rf /', 'dd if=', 'mkfs', ':(){:|:&};:']
        if any(cmd in command for cmd in dangerous_commands):
            return "错误: 检测到危险命令，已拒绝执行"
        
        result = subprocess.run(
            command,
            shell=True,
            cwd=work_path,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        output = f"命令: {command}\n"
        output += f"工作目录: {work_path}\n"
        output += f"返回码: {result.returncode}\n"
        if result.stdout:
            output += f"标准输出:\n{result.stdout}\n"
        if result.stderr:
            output += f"错误输出:\n{result.stderr}\n"
        
        return output
    except subprocess.TimeoutExpired:
        return "错误: 命令执行超时（30秒）"
    except Exception as e:
        return f"执行命令失败: {str(e)}"


# 工具函数映射
TOOLS = {
    "get_current_time": {
        "function": get_current_time,
        "description": "获取当前日期和时间",
        "parameters": []
    },
    "web_search": {
        "function": web_search,
        "description": "搜索网络信息",
        "parameters": ["query"]
    },
    "read_file": {
        "function": read_file,
        "description": "读取文件内容",
        "parameters": ["filepath", "session_id"]
    },
    "write_file": {
        "function": write_file,
        "description": "写入文件内容",
        "parameters": ["filepath", "content", "session_id"]
    },
    "list_files": {
        "function": list_files,
        "description": "列出目录文件",
        "parameters": ["directory", "session_id"]
    },
    "execute_command": {
        "function": execute_command,
        "description": "执行系统命令（编译、运行代码等）",
        "parameters": ["command", "cwd", "session_id"]
    }
}


def parse_tool_calls(message):
    """解析 AI 返回的工具调用"""
    tool_calls = []
    import re
    pattern = r'<tool>(.*?)</tool>'
    matches = re.findall(pattern, message, re.DOTALL)
    
    for match in matches:
        try:
            tool_data = json.loads(match.strip())
            tool_calls.append(tool_data)
        except:
            continue
    
    return tool_calls


def execute_tools(tool_calls, session_id='default'):
    """执行工具调用"""
    results = []
    
    for tool_call in tool_calls:
        tool_name = tool_call.get('name')
        params = tool_call.get('parameters', {})
        
        if tool_name in TOOLS:
            tool_func = TOOLS[tool_name]['function']
            try:
                # 如果工具支持 session_id，添加它
                if 'session_id' in TOOLS[tool_name]['parameters']:
                    params['session_id'] = session_id
                
                result = tool_func(**params)
                results.append({
                    'tool': tool_name,
                    'result': result
                })
            except Exception as e:
                results.append({
                    'tool': tool_name,
                    'result': f"错误: {str(e)}"
                })
    
    return results


# ============================================
# 路由
# ============================================

@app.route('/')
def index():
    return render_template('index.html', models=MODELS, default_work_dir=DEFAULT_WORK_DIR)


@app.route('/api/models')
def get_models():
    return jsonify(MODELS)


@app.route('/api/tools')
def get_tools():
    """返回可用工具列表"""
    tools_info = {}
    for name, info in TOOLS.items():
        tools_info[name] = {
            'description': info['description'],
            'parameters': info['parameters']
        }
    return jsonify(tools_info)


@app.route('/api/project/set', methods=['POST'])
def set_project_path():
    """设置项目路径"""
    data = request.json
    session_id = data.get('session_id', 'default')
    project_path = data.get('project_path', '')
    
    if not project_path:
        return jsonify({"success": False, "error": "项目路径不能为空"})
    
    expanded_path = os.path.expanduser(project_path)
    
    if not os.path.exists(expanded_path):
        return jsonify({"success": False, "error": f"路径不存在: {expanded_path}"})
    
    if not os.path.isdir(expanded_path):
        return jsonify({"success": False, "error": f"不是有效的目录: {expanded_path}"})
    
    user_project_paths[session_id] = expanded_path
    return jsonify({"success": True, "project_path": expanded_path})


@app.route('/api/project/get', methods=['POST'])
def get_project_path():
    """获取当前项目路径"""
    data = request.json
    session_id = data.get('session_id', 'default')
    project_path = user_project_paths.get(session_id, DEFAULT_WORK_DIR)
    return jsonify({"project_path": project_path})


@app.route('/api/chat/stream', methods=['POST'])
def chat_stream():
    """流式聊天（支持工具调用、图片和取消）"""
    data = request.json
    message = data.get('message', '')
    model_key = data.get('model', 'claude-4-sonnet')
    session_id = data.get('session_id', 'default')
    images = data.get('images', [])  # 图片数据（base64）
    
    model = MODELS.get(model_key, model_key)
    
    if session_id not in conversations:
        conversations[session_id] = []
    
    history = conversations[session_id]
    
    # 添加系统提示
    if not history:
        project_path = user_project_paths.get(session_id, DEFAULT_WORK_DIR)
        current_time = get_current_time()
        
        system_prompt = f"""你是一个功能强大的 AI 助手，可以帮助用户完成各种任务。

{current_time}

当前项目路径: {project_path}

你可以使用以下工具：
1. get_current_time() - 获取当前日期和时间（包括星期几）
2. web_search(query) - 搜索网络信息
3. read_file(filepath) - 读取文件内容（相对于项目路径）
4. write_file(filepath, content) - 写入文件
5. list_files(directory) - 列出目录文件
6. execute_command(command, cwd) - 执行命令（编译、运行代码等）

当需要使用工具时，请以以下格式返回：
<tool>{{"name": "tool_name", "parameters": {{"param1": "value1"}}}}</tool>

重要：当用户问时间、日期、星期几时，必须调用 get_current_time() 工具获取准确的当前时间！

你还可以分析用户上传的图片，帮助识别图片内容、代码截图、错误信息等。

你可以在一次回复中调用多个工具。"""
        
        history.append({"role": "system", "content": system_prompt})
    
    # 构建用户消息（包含图片）
    user_message = {"role": "user", "content": []}
    
    # 添加文本内容
    if message:
        user_message["content"].append({"type": "text", "text": message})
    
    # 添加图片内容
    for img_data in images:
        # img_data 格式: data:image/png;base64,xxxxx
        if img_data.startswith('data:image'):
            # 提取 MIME 类型和 base64 数据
            parts = img_data.split(',', 1)
            if len(parts) == 2:
                mime_part = parts[0]  # data:image/png;base64
                base64_data = parts[1]
                
                # 提取 MIME 类型
                media_type = mime_part.split(';')[0].split(':')[1]  # image/png
                
                user_message["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": img_data
                    }
                })
    
    history.append(user_message)
    
    # 标记此请求为活动状态
    request_id = f"{session_id}_{len(history)}"
    active_requests[request_id] = {"cancelled": False}
    
    def generate():
        try:
            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            }
            
            max_iterations = 5
            iteration = 0
            
            while iteration < max_iterations:
                # 检查是否被取消
                if active_requests.get(request_id, {}).get("cancelled"):
                    yield f"data: {json.dumps({'content': '\n\n⏹️ 已停止生成'})}\n\n"
                    yield f"data: {json.dumps({'done': True})}\n\n"
                    break
                
                iteration += 1
                
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
                            # 检查是否被取消
                            if active_requests.get(request_id, {}).get("cancelled"):
                                yield f"data: {json.dumps({'content': '\n\n⏹️ 已停止生成'})}\n\n"
                                yield f"data: {json.dumps({'done': True})}\n\n"
                                return
                            
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
                    
                    # 检查是否有工具调用
                    tool_calls = parse_tool_calls(full_response)
                    
                    if tool_calls:
                        # 执行工具
                        tool_results = execute_tools(tool_calls, session_id)
                        
                        # 发送工具执行结果
                        for result in tool_results:
                            result_msg = f"\n\n🔧 工具执行: {result['tool']}\n{result['result']}\n"
                            yield f"data: {json.dumps({'content': result_msg})}\n\n"
                        
                        # 将结果添加到历史
                        history.append({"role": "assistant", "content": full_response})
                        tool_result_text = "\n".join([r['result'] for r in tool_results])
                        history.append({"role": "user", "content": f"工具执行结果:\n{tool_result_text}\n\n请基于这些结果继续回复用户。"})
                        
                        continue
                    else:
                        # 没有工具调用，结束
                        history.append({"role": "assistant", "content": full_response})
                        yield f"data: {json.dumps({'done': True})}\n\n"
                        break
                        
                except Exception as e:
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
                    break
        finally:
            # 清理请求
            if request_id in active_requests:
                del active_requests[request_id]
    
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )


@app.route('/api/chat/cancel', methods=['POST'])
def cancel_chat():
    """取消当前对话生成"""
    data = request.json
    session_id = data.get('session_id', 'default')
    
    # 标记所有该 session 的请求为取消
    for request_id in list(active_requests.keys()):
        if request_id.startswith(session_id):
            active_requests[request_id]["cancelled"] = True
    
    return jsonify({"success": True})


@app.route('/api/clear', methods=['POST'])
def clear_history():
    """清除对话历史"""
    data = request.json
    session_id = data.get('session_id', 'default')
    
    if session_id in conversations:
        conversations[session_id] = []
    
    return jsonify({"success": True})


if __name__ == '__main__':
    print("🚀 启动增强版 AI 助手...")
    print(f"🔑 API Key: {API_KEY[:20]}...")
    print(f"📁 默认工作目录: {DEFAULT_WORK_DIR}")
    print("📍 访问 http://localhost:5000")
    print("\n可用功能：")
    print("  ✅ 联网搜索")
    print("  ✅ 读写文件")
    print("  ✅ 列出目录")
    print("  ✅ 执行命令")
    print("  ✅ 编译运行代码")
    print("  ✅ 获取准确时间")
    print("  ✅ 设置项目路径")
    print("  ✅ 停止生成")
    print("  ✅ 图片上传识别")
    app.run(host='0.0.0.0', port=5000, debug=True)