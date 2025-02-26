from flask import Flask, request, jsonify, render_template
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# .env 파일 로드
load_dotenv()

app = Flask(__name__)

# 환경 변수에서 Hugging Face API 키 가져오기
HF_API_KEY = os.getenv('HF_API_KEY')
if not HF_API_KEY:
    raise ValueError("Hugging Face API Key가 설정되지 않았습니다.")

# Hugging Face Inference Client 설정
client = InferenceClient(provider="sambanova", api_key=HF_API_KEY)
MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"

# 챗봇 응답 함수
def query(message):
    try:
        messages = [{"role": "user", "content": message}]
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=500,
        )
        return completion.choices[0].message["content"]
    except Exception as e:
        print(f"API 요청 오류: {e}")
        return "⚠️ 응답을 받을 수 없습니다. 서버 오류가 발생했습니다."

# 📌 메인 페이지 렌더링
@app.route('/')
def home():
    return render_template('site.html')

# 📌 AJAX 요청을 처리하는 API 엔드포인트
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get("message", "")
    
    if not user_message.strip():
        return jsonify({"response": "⚠️ 메시지를 입력하세요!"})
    
    bot_message = query(user_message)
    return jsonify({"response": bot_message})

if __name__ == '__main__':
    app.run(debug=True, port=5000)