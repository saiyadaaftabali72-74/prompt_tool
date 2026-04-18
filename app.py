import os
import io
import json
import base64
import sqlite3
from datetime import datetime
from functools import wraps

from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, session, send_file
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from openai import OpenAI

# ENV + APP SETUP
load_dotenv()

BASE_DIR = os.path.abspath(os.path.dirname(_file_))
DB_PATH = os.path.join(BASE_DIR, "prompt_tool.db")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
DOWNLOAD_FILE = os.path.join(BASE_DIR, "prompt_result.txt")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(_name_)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change_me_now_123")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()
TEXT_MODEL = os.environ.get("TEXT_MODEL", "openrouter/auto").strip()
VISION_MODEL = os.environ.get("VISION_MODEL", "openrouter/auto").strip()
SITE_URL = os.environ.get("SITE_URL", "https://prompt-tool-fali.onrender.com").strip()
SITE_NAME = os.environ.get("SITE_NAME", "Prompt Engineer Tool").strip()

print("OPENROUTER KEY FOUND:", bool(OPENROUTER_API_KEY))
print("OPENROUTER KEY PREFIX:", OPENROUTER_API_KEY[:8] if OPENROUTER_API_KEY else "NONE")

# -----------------------------
# OPENROUTER CLIENT
# -----------------------------
def get_openrouter_client():
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY missing on server")

    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        default_headers={
            "HTTP-Referer": SITE_URL,
            "X-OpenRouter-Title": SITE_NAME,
        },
    )


# -----------------------------
# DATABASE
# -----------------------------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS prompts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            mode TEXT,
            model_name TEXT,
            style_name TEXT,
            output_language TEXT,
            idea TEXT,
            result TEXT,
            negative_prompt TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    conn.commit()
    conn.close()


init_db()


# -----------------------------
# HELPERS
# -----------------------------
STYLE_HINTS = {
    "Realistic": "Make it realistic, lifelike, natural, detailed, believable, grounded in real-world visuals.",
    "Ultra-Realistic": "Make it ultra-realistic, highly detailed, premium quality, cinematic realism, realistic textures and lighting.",
    "Cinematic": "Use cinematic framing, dramatic lighting, strong atmosphere, movie-like composition, visual storytelling.",
    "Animated": "Use animation style, clean shapes, expressive visuals, stylized details, vibrant visual design.",
    "3D Animated": "Use 3D animated film style, polished textures, stylized realism, cinematic animated composition.",
    "Anime": "Use anime-inspired style, expressive composition, dramatic mood, strong character and scene detail.",
    "Fantasy": "Use fantasy worldbuilding, magical atmosphere, imaginative details, epic visual storytelling.",
    "Dark": "Use dark moody atmosphere, dramatic shadows, mysterious tone, intense visual emotion.",
    "Minimal": "Keep it clean, simple, elegant, focused, uncluttered, modern.",
}


LANGUAGE_HINTS = {
    "Auto": "Reply in the same language as the user's input. If mixed language is used, respond naturally in that mixed style.",
    "English": "Reply only in English.",
    "Hindi": "Reply only in Hindi.",
    "Hinglish": "Reply only in Hinglish.",
    "Urdu": "Reply only in Urdu.",
    "Arabic": "Reply only in Arabic.",
    "French": "Reply only in French.",
    "Spanish": "Reply only in Spanish.",
    "German": "Reply only in German.",
    "Japanese": "Reply only in Japanese.",
    "Korean": "Reply only in Korean.",
    "Chinese": "Reply only in Chinese.",
}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get("user_id"):
            return jsonify({"error": "Please login first."}), 401
        return fn(*args, **kwargs)
    return wrapper


def save_text_file(content: str):
    with open(DOWNLOAD_FILE, "w", encoding="utf-8") as f:
        f.write(content)


def build_system_prompt(mode: str, model_name: str, style_name: str, output_language: str) -> str:
    style_hint = STYLE_HINTS.get(style_name, STYLE_HINTS["Realistic"])
    language_hint = LANGUAGE_HINTS.get(output_language, LANGUAGE_HINTS["Auto"])

    if mode == "Generate Perfect Prompt":
        task = """
You are an expert prompt engineer.
Your job is to transform a rough user idea into:
1. a strong final generation prompt
2. a matching negative prompt

Return ONLY valid JSON in this format:
{
  "result": "final prompt here",
  "negative_prompt": "negative prompt here"
}
"""
    else:
        task = """
You are an expert creative assistant.
Improve the user's idea and turn it into:
1. a polished final generation prompt
2. a matching negative prompt

Return ONLY valid JSON in this format:
{
  "result": "final prompt here",
  "negative_prompt": "negative prompt here"
}
"""

    model_hint = f"Target model style: {model_name}."
    style_line = f"Visual style guidance: {style_hint}"
    language_line = f"Language guidance: {language_hint}"

    extra = """
The final prompt must be detailed, clean, visually rich, and practical for image generation.
The negative prompt must remove common quality issues, distortions, bad anatomy, blur, text, watermark, low quality, ugly details, and unwanted artifacts.
Do not include markdown.
Return JSON only.
"""

    return "\n".join([task, model_hint, style_line, language_line, extra]).strip()


def call_text_model(system_prompt: str, user_prompt: str) -> dict:
    client = get_openrouter_client()

    response = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.8,
    )

    raw = response.choices[0].message.content.strip()

    try:
        data = json.loads(raw)
        result = (data.get("result") or "").strip()
        negative_prompt = (data.get("negative_prompt") or "").strip()
        return {
            "result": result,
            "negative_prompt": negative_prompt
        }
    except Exception:
        # fallback if model returned text instead of JSON
        return {
            "result": raw,
            "negative_prompt": "low quality, blurry, distorted, bad anatomy, extra fingers, extra limbs, watermark, text, logo, cropped, deformed, ugly"
        }


def image_file_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_vision_model(image_path: str, model_name: str, style_name: str, output_language: str) -> dict:
    client = get_openrouter_client()
    b64 = image_file_to_base64(image_path)
    ext = image_path.rsplit(".", 1)[-1].lower()
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"

    style_hint = STYLE_HINTS.get(style_name, STYLE_HINTS["Realistic"])
    language_hint = LANGUAGE_HINTS.get(output_language, LANGUAGE_HINTS["Auto"])

    system_prompt = f"""
You are an expert image-to-prompt converter.
Look at the image and generate:
1. a strong image generation prompt
2. a matching negative prompt

Language guidance: {language_hint}
Style guidance: {style_hint}

Return ONLY valid JSON in this format:
{{
  "result": "final prompt here",
  "negative_prompt": "negative prompt here"
}}
"""

    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Convert this image into a strong {model_name} prompt."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"}
                    },
                ],
            },
        ],
        temperature=0.5,
    )

    raw = response.choices[0].message.content.strip()

    try:
        data = json.loads(raw)
        return {
            "result": (data.get("result") or "").strip(),
            "negative_prompt": (data.get("negative_prompt") or "").strip()
        }
    except Exception:
        return {
            "result": raw,
            "negative_prompt": "low quality, blurry, distorted, bad anatomy, extra fingers, extra limbs, watermark, text, logo, cropped, deformed, ugly"
        }


def generate_prompt_with_ai(idea: str, mode: str, model_name: str, style_name: str, output_language: str) -> dict:
    system_prompt = build_system_prompt(mode, model_name, style_name, output_language)
    return call_text_model(system_prompt, idea)


# -----------------------------
# ROUTES
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/me", methods=["GET"])
def me():
    if not session.get("user_id"):
        return jsonify({"logged_in": False})

    return jsonify({
        "logged_in": True,
        "user": {
            "id": session.get("user_id"),
            "username": session.get("username"),
            "email": session.get("email"),
        }
    })


@app.route("/api/signup", methods=["POST"])
def signup():
    data = request.get_json()
    username = (data.get("username") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = (data.get("password") or "").strip()

    if not username or not email or not password:
        return jsonify({"error": "Username, email, and password are required."}), 400

    conn = get_db()
    cur = conn.cursor()

    existing = cur.execute(
        "SELECT id FROM users WHERE email = ? OR username = ?",
        (email, username)
    ).fetchone()

    if existing:
        conn.close()
        return jsonify({"error": "User already exists."}), 400

    password_hash = generate_password_hash(password)
    cur.execute(
        "INSERT INTO users (username, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
        (username, email, password_hash, datetime.utcnow().isoformat())
    )
    conn.commit()

    user = cur.execute(
        "SELECT id, username, email FROM users WHERE email = ?",
        (email,)
    ).fetchone()

    conn.close()

    session["user_id"] = user["id"]
    session["username"] = user["username"]
    session["email"] = user["email"]

    return jsonify({"message": "Signup successful."})


@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json()
    email = (data.get("email") or "").strip().lower()
    password = (data.get("password") or "").strip()

    if not email or not password:
        return jsonify({"error": "Email and password are required."}), 400

    conn = get_db()
    user = conn.execute(
        "SELECT * FROM users WHERE email = ?",
        (email,)
    ).fetchone()
    conn.close()

    if not user or not check_password_hash(user["password_hash"], password):
        return jsonify({"error": "Invalid email or password."}), 401

    session["user_id"] = user["id"]
    session["username"] = user["username"]
    session["email"] = user["email"]

    return jsonify({"message": "Login successful."})


@app.route("/api/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"message": "Logged out successfully."})


@app.route("/api/process", methods=["POST"])
def process():
    if not OPENROUTER_API_KEY:
        return jsonify({
            "result": "OPENROUTER_API_KEY missing on server",
            "negative_prompt": "OPENROUTER_API_KEY missing on server"
        }), 500

    data = request.get_json()

    idea = (data.get("idea") or "").strip()
    mode = (data.get("mode") or "Generate Perfect Prompt").strip()
    model_name = (data.get("model_name") or "Flux").strip()
    style_name = (data.get("style_name") or "Realistic").strip()
    output_language = (data.get("output_language") or "Auto").strip()

    if not idea:
        return jsonify({"error": "Please enter your idea first."}), 400

    try:
        result = generate_prompt_with_ai(idea, mode, model_name, style_name, output_language)
        return jsonify(result)
    except Exception as e:
        return jsonify({
            "result": f"Error code: 500 - {str(e)}",
            "negative_prompt": f"Error code: 500 - {str(e)}"
        }), 500


@app.route("/api/image-to-prompt", methods=["POST"])
def image_to_prompt():
    if not OPENROUTER_API_KEY:
        return jsonify({"error": "OPENROUTER_API_KEY missing on server"}), 500

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    image = request.files["image"]
    model_name = request.form.get("model_name", "Flux")
    style_name = request.form.get("style_name", "Realistic")
    output_language = request.form.get("output_language", "Auto")

    if image.filename == "":
        return jsonify({"error": "No selected file."}), 400

    if not allowed_file(image.filename):
        return jsonify({"error": "Unsupported file type."}), 400

    filename = secure_filename(image.filename)
    unique_name = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{filename}"
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    image.save(file_path)

    session["last_uploaded_image"] = file_path

    try:
        result = call_vision_model(file_path, model_name, style_name, output_language)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Error code: 500 - {str(e)}"}), 500


@app.route("/api/delete-uploaded-image", methods=["POST"])
def delete_uploaded_image():
    file_path = session.get("last_uploaded_image")

    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception:
            pass

    session.pop("last_uploaded_image", None)
    return jsonify({"message": "Uploaded image deleted."})


@app.route("/api/save-prompt", methods=["POST"])
@login_required
def save_prompt():
    data = request.get_json()

    mode = (data.get("mode") or "").strip()
    model_name = (data.get("model_name") or "").strip()
    style_name = (data.get("style_name") or "").strip()
    output_language = (data.get("output_language") or "").strip()
    idea = (data.get("idea") or "").strip()
    result = (data.get("result") or "").strip()
    negative_prompt = (data.get("negative_prompt") or "").strip()

    if not result:
        return jsonify({"error": "Nothing to save."}), 400

    conn = get_db()
    conn.execute("""
        INSERT INTO prompts (
            user_id, mode, model_name, style_name, output_language,
            idea, result, negative_prompt, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        session["user_id"], mode, model_name, style_name, output_language,
        idea, result, negative_prompt, datetime.utcnow().isoformat()
    ))
    conn.commit()
    conn.close()

    return jsonify({"message": "Prompt saved successfully."})


@app.route("/api/prompts", methods=["GET"])
@login_required
def get_prompts():
    conn = get_db()
    rows = conn.execute("""
        SELECT id, mode, model_name, style_name, output_language, idea,
               result, negative_prompt, created_at
        FROM prompts
        WHERE user_id = ?
        ORDER BY id DESC
    """, (session["user_id"],)).fetchall()
    conn.close()

    prompts = [dict(row) for row in rows]
    return jsonify({"prompts": prompts})


@app.route("/api/download-txt", methods=["POST"])
def download_txt():
    data = request.get_json()
    result = (data.get("result") or "").strip()
    negative_prompt = (data.get("negative_prompt") or "").strip()

    content = f"Generated Prompt:\n{result}\n\nNegative Prompt:\n{negative_prompt}\n"
    save_text_file(content)

    return send_file(
        DOWNLOAD_FILE,
        as_attachment=True,
        download_name="prompt_result.txt",
        mimetype="text/plain"
    )


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
