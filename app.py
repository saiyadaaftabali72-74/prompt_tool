import os
import sqlite3
import base64
from datetime import datetime
from functools import wraps

from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, session
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from openai import OpenAI

load_dotenv()

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "prompt_tool.db")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "change_me_now")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

OPENROUTER_API_KEY = "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".strip()
TEXT_MODEL = os.getenv("TEXT_MODEL", "openrouter/auto")
VISION_MODEL = os.getenv("VISION_MODEL", "openrouter/free")
SITE_URL = os.getenv("SITE_URL", "http://127.0.0.1:5000")
SITE_NAME = os.getenv("SITE_NAME", "Prompt Engineer Tool")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    default_headers={
        "HTTP-Referer": SITE_URL,
        "X-OpenRouter-Title": SITE_NAME,
    },
)

MODEL_PROMPT_HINTS = {
    "Midjourney": "Optimize for Midjourney style prompt writing. Use descriptive visual language and a strong artistic composition.",
    "Flux": "Optimize for Flux image generation. Keep the prompt clear, rich, and visually precise.",
    "SDXL": "Optimize for SDXL prompt writing. Use strong visual descriptors and photorealistic details where relevant.",
}

MODE_HINTS = {
    "Image Prompt": "Create one polished image-generation prompt.",
    "Video Prompt": "Create one cinematic video-generation prompt with motion, camera movement, atmosphere, and scene flow.",
    "Logo Prompt": "Create one clean logo-design prompt with brand identity, symbolism, composition, and visual simplicity.",
    "Product Ad Prompt": "Create one premium ad-style prompt that highlights the product, background, lighting, and commercial quality.",
    "Thumbnail Prompt": "Create one bold, high-click-through thumbnail prompt with visual contrast, readability, and dramatic composition.",
}

STYLE_HINTS = {
    "Cinematic": "Use cinematic, dramatic, visually rich language.",
    "Realistic": "Keep it realistic, natural, highly detailed, and believable.",
    "Luxury": "Use premium, elegant, polished, expensive-looking aesthetics.",
    "Fantasy": "Use magical, mythical, epic visual language.",
    "Anime": "Use anime-inspired style, expressive visuals, and strong stylization.",
    "Fashion": "Use editorial, stylish, trendy, magazine-quality aesthetics.",
    "Horror": "Use dark, eerie, unsettling atmosphere and visual details.",
    "Advertising": "Use commercial, polished, high-conversion ad visuals.",
}

NEGATIVE_BY_MODE = {
    "Image Prompt": "blurry, low quality, bad anatomy, extra fingers, deformed hands, distorted face, watermark, text, cropped, noisy, oversaturated",
    "Video Prompt": "shaky camera, blurry motion, low quality, flicker, bad transitions, artifacts, distorted body, watermark, text overlay",
    "Logo Prompt": "blurry, cluttered design, low resolution, bad typography, distorted shapes, watermark, noisy background",
    "Product Ad Prompt": "blurry, poor lighting, messy background, distorted product, low quality, watermark, text overlay",
    "Thumbnail Prompt": "blurry, dull colors, cluttered composition, weak contrast, unreadable subject, watermark, low quality",
}

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
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS saved_prompts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            idea TEXT NOT NULL,
            mode TEXT NOT NULL,
            style TEXT NOT NULL,
            model_label TEXT NOT NULL,
            prompt TEXT NOT NULL,
            negative_prompt TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    conn.commit()
    conn.close()

def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return jsonify({"error": "Please login first."}), 401
        return fn(*args, **kwargs)
    return wrapper

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def model_label_to_api_name(model_label: str) -> str:
    # UI labels map to one configured text model for simplicity.
    # You can later map each one to different OpenRouter models if you want.
    return TEXT_MODEL

def build_system_prompt(mode: str, style: str, model_label: str, action: str) -> str:
    mode_hint = MODE_HINTS.get(mode, MODE_HINTS["Image Prompt"])
    style_hint = STYLE_HINTS.get(style, STYLE_HINTS["Cinematic"])
    model_hint = MODEL_PROMPT_HINTS.get(model_label, MODEL_PROMPT_HINTS["Midjourney"])

    action_hint = {
        "generate": "Turn the user's short idea into one excellent final prompt.",
        "shorten": "Shorten the prompt while keeping it strong, clear, and useful.",
        "improve_more": "Take the user's idea and make the prompt much more detailed, refined, and visually rich.",
    }.get(action, "Turn the user's short idea into one excellent final prompt.")

    return f"""
You are a world-class AI prompt engineer.

Rules:
- Always respond in English.
- Return only the final prompt. No explanation, no bullets, no notes.
- Make the result production-ready.
- Include subject, setting, mood, lighting, composition, texture, style, and quality cues when relevant.

Mode guidance:
{mode_hint}

Style guidance:
{style_hint}

Model guidance:
{model_hint}

Task:
{action_hint}
""".strip()

def negative_prompt_for_mode(mode: str) -> str:
    return NEGATIVE_BY_MODE.get(mode, NEGATIVE_BY_MODE["Image Prompt"])

def ensure_api_key():
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is missing in .env")

def generate_text_prompt(user_input: str, mode: str, style: str, model_label: str, action: str) -> str:
    ensure_api_key()

    if not user_input.strip():
        return "Please enter an idea first."

    response = client.chat.completions.create(
        model=model_label_to_api_name(model_label),
        messages=[
            {"role": "system", "content": build_system_prompt(mode, style, model_label, action)},
            {"role": "user", "content": user_input},
        ],
    )

    return response.choices[0].message.content.strip()

def image_file_to_data_url(filepath: str) -> str:
    ext = os.path.splitext(filepath)[1].lower().replace(".", "")
    if ext == "jpg":
        ext = "jpeg"
    mime = f"image/{ext}"
    with open(filepath, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"

def generate_image_prompt_from_uploaded_image(filepath: str, mode: str, style: str, model_label: str) -> str:
    ensure_api_key()

    image_data_url = image_file_to_data_url(filepath)

    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a world-class prompt engineer for AI image generation. "
                    "Analyze the uploaded image and convert it into one polished, detailed, visually rich prompt in English. "
                    "Return only the final prompt. No explanation."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Mode: {mode}\nStyle: {style}\nTarget style: {model_label}\nCreate one perfect generation prompt from this image."},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
    )

    return response.choices[0].message.content.strip()

@app.route("/")
def home():
    return render_template("index.html", username=session.get("username"))

@app.route("/api/signup", methods=["POST"])
def signup():
    data = request.get_json(force=True)
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()

    if len(username) < 3 or len(password) < 6:
        return jsonify({"error": "Username must be 3+ chars and password 6+ chars."}), 400

    password_hash = generate_password_hash(password)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        conn = get_db()
        conn.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, password_hash, now),
        )
        conn.commit()
        conn.close()
        return jsonify({"message": "Signup successful. Please login."})
    except sqlite3.IntegrityError:
        return jsonify({"error": "Username already exists."}), 400

@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json(force=True)
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()

    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()

    if not user or not check_password_hash(user["password_hash"], password):
        return jsonify({"error": "Invalid username or password."}), 401

    session["user_id"] = user["id"]
    session["username"] = user["username"]
    return jsonify({"message": "Login successful.", "username": user["username"]})

@app.route("/api/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"message": "Logged out."})

@app.route("/api/me", methods=["GET"])
def me():
    return jsonify({
        "logged_in": "user_id" in session,
        "username": session.get("username")
    })

@app.route("/api/process", methods=["POST"])
def process():
    try:
        data = request.get_json(force=True)
        text = data.get("text", "").strip()
        mode = data.get("mode", "Image Prompt")
        style = data.get("style", "Cinematic")
        model_label = data.get("model_label", "Midjourney")
        action = data.get("action", "generate")

        prompt = generate_text_prompt(text, mode, style, model_label, action)
        negative = negative_prompt_for_mode(mode)

        return jsonify({
            "prompt": prompt,
            "negative_prompt": negative
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/image-to-prompt", methods=["POST"])
def image_to_prompt():
    try:
        if "image" not in request.files:
            return jsonify({"error": "Please upload an image."}), 400

        file = request.files["image"]
        mode = request.form.get("mode", "Image Prompt")
        style = request.form.get("style", "Cinematic")
        model_label = request.form.get("model_label", "Midjourney")

        if file.filename == "":
            return jsonify({"error": "No file selected."}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "Only png, jpg, jpeg, webp files are allowed."}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        prompt = generate_image_prompt_from_uploaded_image(filepath, mode, style, model_label)
        negative = negative_prompt_for_mode(mode)

        return jsonify({
            "prompt": prompt,
            "negative_prompt": negative
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/save-prompt", methods=["POST"])
@login_required
def save_prompt():
    data = request.get_json(force=True)

    idea = data.get("idea", "").strip()
    mode = data.get("mode", "").strip()
    style = data.get("style", "").strip()
    model_label = data.get("model_label", "").strip()
    prompt = data.get("prompt", "").strip()
    negative_prompt = data.get("negative_prompt", "").strip()

    if not prompt:
        return jsonify({"error": "Nothing to save."}), 400

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = get_db()
    conn.execute("""
        INSERT INTO saved_prompts (user_id, idea, mode, style, model_label, prompt, negative_prompt, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (session["user_id"], idea, mode, style, model_label, prompt, negative_prompt, now))
    conn.commit()
    conn.close()

    return jsonify({"message": "Prompt saved."})

@app.route("/api/saved-prompts", methods=["GET"])
@login_required
def get_saved_prompts():
    q = request.args.get("q", "").strip()

    conn = get_db()
    if q:
        rows = conn.execute("""
            SELECT * FROM saved_prompts
            WHERE user_id = ?
              AND (
                idea LIKE ?
                OR mode LIKE ?
                OR style LIKE ?
                OR model_label LIKE ?
                OR prompt LIKE ?
                OR negative_prompt LIKE ?
              )
            ORDER BY id DESC
        """, (
            session["user_id"],
            f"%{q}%", f"%{q}%", f"%{q}%", f"%{q}%", f"%{q}%", f"%{q}%"
        )).fetchall()
    else:
        rows = conn.execute("""
            SELECT * FROM saved_prompts
            WHERE user_id = ?
            ORDER BY id DESC
        """, (session["user_id"],)).fetchall()

    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route("/api/saved-prompts/<int:prompt_id>", methods=["DELETE"])
@login_required
def delete_saved_prompt(prompt_id: int):
    conn = get_db()
    conn.execute(
        "DELETE FROM saved_prompts WHERE id = ? AND user_id = ?",
        (prompt_id, session["user_id"])
    )
    conn.commit()
    conn.close()
    return jsonify({"message": "Saved prompt deleted."})

if __name__ == "__main__":
    init_db()
    app.run(debug=True)
