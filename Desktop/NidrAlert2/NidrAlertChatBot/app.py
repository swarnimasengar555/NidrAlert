from flask import Flask, render_template, request, jsonify, session 
from flask_cors import CORS
from groq import Groq
import uuid 
import os
from dotenv import load_dotenv

load_dotenv() 

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY")

# ── CORS: allow your React app to send cookies ──────────────────────
CORS(app, supports_credentials=True, origins=[
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
    # add your deployed frontend URL here when you go live e.g.:
    # "https://your-app.vercel.app",
])

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
print("groq is all set")

SYSTEM_PROMPT = """You are NidrAlert, an expert AI assistant specializing in Indian road safety 
and drowsy/drunk driving prevention. You help users with:
- Indian traffic rules and road safety laws (Motor Vehicles Act)
- Dangers of drowsy driving and how to stay alert on Indian roads
- Safe driving practices on Indian highways, expressways, and city roads
- Emergency procedures and helpline numbers in India (like 112, 1033)
- Warning signs of fatigue while driving
- Best rest stop practices for long-distance drivers in India
Always respond in a helpful, clear, and safety-focused manner. 
If asked anything unrelated to road safety or drowsy driving, 
politely redirect the user back to road safety topics."""

conversation_store = {} 

@app.route("/")
def home():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        # For API calls (no browser session), use a header or body session_id
        session_id = session.get("session_id")
        if not session_id:
            session_id = request.json.get("session_id", str(uuid.uuid4()))
            session["session_id"] = session_id

        data = request.get_json()
        user_message = data.get("message", "").strip()

        if not user_message:
            return jsonify({"error": "empty message"}), 400

        if session_id not in conversation_store:  
            conversation_store[session_id] = []   

        history = conversation_store[session_id]
        history.append({"role": "user", "content": user_message})

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *history
            ],
            max_tokens=512,
            temperature=0.7,
        )

        assistant_reply = response.choices[0].message.content
        history.append({"role": "assistant", "content": assistant_reply})
        conversation_store[session_id] = history

        return jsonify({"reply": assistant_reply})
    except Exception as e:
        print(f"ERROR in /chat route: {e}")
        return jsonify({"reply": f"Server error: {str(e)}"}), 500

@app.route("/clear", methods=["POST"])
def clear():
    session_id = session.get("session_id")
    if session_id and session_id in conversation_store:
        conversation_store[session_id] = []  
    return jsonify({"status": "cleared"})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)