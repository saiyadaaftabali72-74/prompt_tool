import os
import json
import sqlite3
import base64
from datetime import datetime
from functools import wraps

from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, session, send_file
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from openai import OpenAI

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(ENV_PATH)

DB_PATH = os.path.join(BASE_DIR, "prompt_tool.db")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
DOWNLOAD_FILE = os.path.join(BASE_DIR, "prompt_result.txt")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "change_me_now_123")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

OPENROUTER_API_KEY = "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"
TEXT_MODEL = os.getenv("TEXT_MODEL", "openrouter/auto")
VISION_MODEL = os.getenv("VISION_MODEL", "openrouter/auto")
SITE_URL = os.getenv("SITE_URL", "http://127.0.0.1:5000")
SITE_NAME = os.getenv("SITE_NAME", "Prompt Engineer Tool")

print("ENV PATH:", ENV_PATH)
print("OPENROUTER KEY FOUND:", bool(OPENROUTER_API_KEY))

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    default_headers={
        "HTTP-Referer": SITE_URL,
        "X-OpenRouter-Title": SITE_NAME,
    },
)

STYLE_HINTS = {
    "Ultra-Realistic": "Make the prompt ultra-realistic, lifelike, richly detailed, highly photographic, with natural lighting, premium textures, and realism.",
    "Realistic": "Make the prompt realistic, believable, visually grounded, and naturally detailed.",
    "Animated": "Make the prompt suitable for high-quality animated visuals with expressive design and appealing stylization.",
    "Cinematic": "Make the prompt cinematic with dramatic composition, mood, camera language, depth, and film-style lighting.",
    "Fantasy": "Make the prompt imaginative, magical, epic, and visually rich.",
    "Anime": "Make the prompt suitable for anime-style visuals with expressive characters and stylized details.",
    "3D Render": "Make the prompt suitable for a polished 3D render with realistic materials and strong lighting.",
    "Pixel Art": "Make the prompt suitable for pixel art with crisp retro detail and readable composition."
}

MODEL_HINTS = {
    "Flux": "Optimize for Flux image generation. Keep the prompt visually clear, rich, modern, and precise.",
    "Midjourney": "Optimize for Midjourney with elegant visual phrasing, stylization, artistic composition, and strong direction.",
    "SDXL": "Optimize for SDXL with detailed descriptors, coherent scene design, and clear subject emphasis."
}

LANGUAGE_HINTS = {
    "Auto": "Understand the user's language automatically and respond in the most suitable language.",
    "English": "Write the final output in natural English.",
    "Hindi": "Write the final output in natural Hindi.",
    "Urdu": "Write the final output in natural Urdu.",
    "Arabic": "Write the final output in natural Arabic.",
    "Spanish": "Write the final output in natural Spanish.",
    "French": "Write the final output in natural French.",
    "German": "Write the final output in natural German.",
    "Portuguese": "Write the final output in natural Portuguese.",
    "Russian": "Write the final output in natural Russian.",
    "Japanese": "Write the final output in natural Japanese.",
    "Korean": "Write the final output in natural Korean.",
    "Chinese": "Write the final output in natural Chinese."
}


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_column(cur, table_name, column_name, column_def):
    cur.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in cur.fetchall()]
    if column_name not in columns:
        cur.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_def}")


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
        CREATE TABLE IF NOT EXISTS saved_prompts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            idea TEXT NOT NULL,
            final_prompt TEXT NOT NULL,
            mode TEXT NOT NULL,
            model_name TEXT NOT NULL,
            style_name TEXT NOT NULL,
            output_language TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    ensure_column(cur, "saved_prompts", "negative_prompt", "negative_prompt TEXT DEFAULT ''")

    conn.commit()
    conn.close()


def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return jsonify({"error": "Please login first."}), 401
        return fn(*args, **kwargs)
    return wrapper


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def build_system_prompt(mode, model_name, style_name, output_language):
    model_hint = MODEL_HINTS.get(model_name, MODEL_HINTS["Flux"])
    style_hint = STYLE_HINTS.get(style_name, STYLE_HINTS["Realistic"])
    language_hint = LANGUAGE_HINTS.get(output_language, LANGUAGE_HINTS["Auto"])

    base = (
        "You are an elite prompt engineer. "
        "You understand casual user wording, mixed-language chat, imperfect spelling, and Hindi written in Roman letters. "
        "Understand what the user means and convert it into a polished professional result. "
        f"{model_hint} {style_hint} {language_hint} "
    )

    if mode == "generate":
        return base + (
            "Create a strong positive prompt and also a useful negative prompt. "
            "Improve subject details, environment, composition, mood, camera, lighting, style, colors, and useful visual details."
        )
    if mode == "improve":
        return base + (
            "Improve the user's prompt strongly and also provide a suitable negative prompt."
        )
    if mode == "shorten":
        return base + (
            "Shorten the user's prompt while preserving the meaning, and also provide a suitable negative prompt."
        )
    return base


def safe_prompt_json(content):
    try:
        data = json.loads(content)
        return {
            "positive_prompt": (data.get("positive_prompt") or "").strip(),
            "negative_prompt": (data.get("negative_prompt") or "").strip()
        }
    except Exception:
        return {
            "positive_prompt": content.strip(),
            "negative_prompt": "low quality, blurry, bad anatomy, extra fingers, distorted, watermark, text, cropped, duplicate, noisy, ugly"
        }


def generate_prompt_with_ai(idea, mode, model_name, style_name, output_language):
    system_prompt = build_system_prompt(mode, model_name, style_name, output_language)

    user_prompt = (
        f"User input:\n{idea}\n\n"
        "Return JSON only in this exact format:\n"
        '{'
        '"positive_prompt":"...",'
        '"negative_prompt":"..."'
        '}\n'
        "Do not write any explanation outside JSON."
    )

    response = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.9
    )

    return safe_prompt_json(response.choices[0].message.content.strip())


def image_to_prompt_with_ai(image_path, model_name, style_name, output_language):
    model_hint = MODEL_HINTS.get(model_name, MODEL_HINTS["Flux"])
    style_hint = STYLE_HINTS.get(style_name, STYLE_HINTS["Realistic"])
    language_hint = LANGUAGE_HINTS.get(output_language, LANGUAGE_HINTS["Auto"])

    ext = os.path.splitext(image_path)[1].lower()
    mime = "image/png"
    if ext in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    elif ext == ".webp":
        mime = "image/webp"

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    data_url = f"data:{mime};base64,{b64}"

    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert visual prompt engineer. "
                    "Analyze the uploaded image and create a polished positive prompt and a useful negative prompt. "
                    f"{model_hint} {style_hint} {language_hint} "
                    'Return JSON only in this format: {"positive_prompt":"...","negative_prompt":"..."}'
                )
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Create a strong AI image prompt and negative prompt from this uploaded image."},
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]
            }
        ],
        temperature=0.7
    )

    return safe_prompt_json(response.choices[0].message.content.strip())


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/me", methods=["GET"])
def api_me():
    if "user_id" not in session:
        return jsonify({"logged_in": False})

    return jsonify({
        "logged_in": True,
        "user": {
            "id": session["user_id"],
            "username": session["username"],
            "email": session["email"]
        }
    })


@app.route("/api/signup", methods=["POST"])
def signup():
    data = request.get_json()
    username = (data.get("username") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not username or not email or not password:
        return jsonify({"error": "All signup fields are required."}), 400

    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters."}), 400

    conn = get_db()
    cur = conn.cursor()

    try:
        cur.execute(
            "INSERT INTO users (username, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (username, email, generate_password_hash(password), datetime.utcnow().isoformat())
        )
        conn.commit()
        user_id = cur.lastrowid

        session["user_id"] = user_id
        session["username"] = username
        session["email"] = email

        return jsonify({"message": "Signup successful."})
    except sqlite3.IntegrityError:
        return jsonify({"error": "Username or email already exists."}), 400
    finally:
        conn.close()


@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE email = ?", (email,))
    user = cur.fetchone()
    conn.close()

    if not user:
        return jsonify({"error": "User not found."}), 404

    if not check_password_hash(user["password_hash"], password):
        return jsonify({"error": "Wrong password."}), 401

    session["user_id"] = user["id"]
    session["username"] = user["username"]
    session["email"] = user["email"]

    return jsonify({"message": "Login successful."})


@app.route("/api/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"message": "Logged out successfully."})


@app.route("/api/process", methods=["POST"])
def process_prompt():
    if not OPENROUTER_API_KEY:
        return jsonify({"error": "OPENROUTER_API_KEY is missing in .env"}), 500

    data = request.get_json()
    idea = (data.get("idea") or "").strip()
    mode = (data.get("mode") or "generate").strip()
    model_name = (data.get("model_name") or "Flux").strip()
    style_name = (data.get("style_name") or "Realistic").strip()
    output_language = (data.get("output_language") or "Auto").strip()

    if not idea:
        return jsonify({"error": "Please enter your idea first."}), 400

    try:
        result = generate_prompt_with_ai(idea, mode, model_name, style_name, output_language)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/image-to-prompt", methods=["POST"])
def image_to_prompt():
    if not OPENROUTER_API_KEY:
        return jsonify({"error": "OPENROUTER_API_KEY is missing in .env"}), 500

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
    save_path = os.path.join(UPLOAD_FOLDER, unique_name)
    image.save(save_path)

    try:
        result = image_to_prompt_with_ai(save_path, model_name, style_name, output_language)
        result["uploaded_filename"] = unique_name
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/delete-uploaded-image", methods=["POST"])
def delete_uploaded_image():
    data = request.get_json()
    filename = (data.get("filename") or "").strip()

    if not filename:
        return jsonify({"error": "No filename provided."}), 400

    safe_name = os.path.basename(filename)
    file_path = os.path.join(UPLOAD_FOLDER, safe_name)

    if os.path.exists(file_path):
        os.remove(file_path)
        return jsonify({"message": "Uploaded image deleted successfully."})

    return jsonify({"error": "File not found."}), 404


@app.route("/api/save-prompt", methods=["POST"])
@login_required
def save_prompt():
    data = request.get_json()

    idea = (data.get("idea") or "").strip()
    final_prompt = (data.get("final_prompt") or "").strip()
    negative_prompt = (data.get("negative_prompt") or "").strip()
    mode = (data.get("mode") or "generate").strip()
    model_name = (data.get("model_name") or "Flux").strip()
    style_name = (data.get("style_name") or "Realistic").strip()
    output_language = (data.get("output_language") or "Auto").strip()

    if not final_prompt:
        return jsonify({"error": "Nothing to save."}), 400

    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO saved_prompts
        (user_id, idea, final_prompt, negative_prompt, mode, model_name, style_name, output_language, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        session["user_id"],
        idea,
        final_prompt,
        negative_prompt,
        mode,
        model_name,
        style_name,
        output_language,
        datetime.utcnow().isoformat()
    ))
    conn.commit()
    conn.close()

    return jsonify({"message": "Prompt saved successfully."})


@app.route("/api/saved-prompts", methods=["GET"])
@login_required
def saved_prompts():
    search = (request.args.get("search") or "").strip()

    conn = get_db()
    cur = conn.cursor()

    if search:
        cur.execute("""
            SELECT * FROM saved_prompts
            WHERE user_id = ?
            AND (
                idea LIKE ?
                OR final_prompt LIKE ?
                OR negative_prompt LIKE ?
                OR model_name LIKE ?
                OR style_name LIKE ?
                OR output_language LIKE ?
            )
            ORDER BY id DESC
        """, (
            session["user_id"],
            f"%{search}%",
            f"%{search}%",
            f"%{search}%",
            f"%{search}%",
            f"%{search}%",
            f"%{search}%"
        ))
    else:
        cur.execute("""
            SELECT * FROM saved_prompts
            WHERE user_id = ?
            ORDER BY id DESC
        """, (session["user_id"],))

    rows = cur.fetchall()
    conn.close()

    items = []
    for row in rows:
        items.append({
            "id": row["id"],
            "idea": row["idea"],
            "final_prompt": row["final_prompt"],
            "negative_prompt": row["negative_prompt"] if "negative_prompt" in row.keys() else "",
            "mode": row["mode"],
            "model_name": row["model_name"],
            "style_name": row["style_name"],
            "output_language": row["output_language"],
            "created_at": row["created_at"]
        })

    return jsonify({"items": items})


@app.route("/api/delete-prompt/<int:prompt_id>", methods=["DELETE"])
@login_required
def delete_prompt(prompt_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("DELETE FROM saved_prompts WHERE id = ? AND user_id = ?", (prompt_id, session["user_id"]))
    conn.commit()
    conn.close()
    return jsonify({"message": "Prompt deleted successfully."})


@app.route("/api/delete-all-history", methods=["DELETE"])
@login_required
def delete_all_history():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("DELETE FROM saved_prompts WHERE user_id = ?", (session["user_id"],))
    conn.commit()
    conn.close()
    return jsonify({"message": "All history deleted successfully."})


@app.route("/api/download-txt", methods=["POST"])
def download_txt():
    data = request.get_json()
    content = (data.get("content") or "").strip()

    if not content:
        return jsonify({"error": "Nothing to download."}), 400

    with open(DOWNLOAD_FILE, "w", encoding="utf-8") as f:
        f.write(content)

    return send_file(DOWNLOAD_FILE, as_attachment=True)


if __name__ == "__main__":
    init_db()
    app.run(debug=True)
