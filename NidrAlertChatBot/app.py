from flask import Flask, render_template, request, jsonify, session 
from groq import Groq
import uuid 
import os
from dotenv import load_dotenv


load_dotenv() 

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY")

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
print("groq is all set")

# ─────────────────────────────────────────────
# SYSTEM PROMPT — Defines the chatbot's personality & specialization
# This message is always sent first (hidden from the user)
# to instruct the AI on what role to play
# ─────────────────────────────────────────────
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

#storing user chats in a dictionary
conversation_store = {} 
#render you to index.html
#route 1 : home page
@app.route("/")
def home():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return render_template("index.html")

#route 2 : chat endpoint
@app.route("/chat", methods=["POST"])
def chat():
    try:

        session_id = session.get("session_id", str(uuid.uuid4()))

        #extract user message
        data = request.get_json()
        user_message = data.get("message", "").strip()

        #don't process empty message
        if not user_message:
            return jsonify({"error": "empty message"}), 400

        #retrive or create chat history
        #if the user is new start fresh
        if session_id not in conversation_store:  
            conversation_store[session_id] = []   

        history = conversation_store[session_id]

        #add new users message to the history
        history.append({"role": "user", "content": user_message})

        # now we send the conversation to Groq's servers — they run it on their GPU (~2 seconds per reply)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},  # personality instructions sent first
                *history  # full conversation so far — this is what gives the bot memory
            ],
            max_tokens=512,
            temperature=0.7,  #creativity level
        )

        # extract just the text reply from Groq's response object
        assistant_reply = response.choices[0].message.content

        #save assistant replay to history , this reply will be included as context
        history.append({"role": "assistant", "content": assistant_reply})

        #save updated history back to the store
        conversation_store[session_id] = history

        #sends the reply back to the frontend
        return jsonify({"reply": assistant_reply})
    except Exception as e:
        print(f"ERROR in /chat route: {e}")  # this will print exact error in terminal
        return jsonify({"reply": f"Server error: {str(e)}"}), 500
#route 3: clear chat history
#called when user clicks on new chat button(wipes convo+session)
@app.route("/clear", methods=["POST"])
def clear():
    session_id = session.get("session_id")
    if session_id and session_id in conversation_store:
        conversation_store[session_id] = []  
    return jsonify({"status": "cleared"})

#start the server
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)