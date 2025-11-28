# app.py  -- upgraded from your original with RAG-ish retrieval + Gemini prompt templates + link extraction
import os
import json
import re
import webbrowser
from collections import Counter
from difflib import get_close_matches
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import PyPDF2
import docx2txt


def open_in_browser(link):
    """Open a link in a new browser tab safely."""
    if not link:
        return False
    try:
        if not link.startswith("http"):
            link = "https://" + link
        webbrowser.open_new_tab(link)
        return True
    except Exception as e:
        print("Browser open failed:", e)
        return False

# Gemini imports (same pattern you used)
try:
    from google import genai
except ImportError:
    genai = None

load_dotenv()
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "projects"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

PROJECT_FILE = os.path.join(app.config["UPLOAD_FOLDER"], "project_data.json")
PROMPT_CONFIG = os.path.join(app.config["UPLOAD_FOLDER"], "prompt_config.json")
CHAT_HISTORY_FILE = "chat_history.json"
RESUME_FILE = os.path.join(app.config["UPLOAD_FOLDER"], "resume_data.json")

# Ensure files exist
if not os.path.exists(PROJECT_FILE):
    with open(PROJECT_FILE, "w") as f:
        json.dump({}, f)
"""
if not os.path.exists(RESUME_FILE):
    with open(RESUME_FILE, "w") as f:
        json.dump({}, f)

# default prompt config (adjustable via /config)
default_prompt_cfg = {
    "resume_extraction_prompt": "You are a precise resume extractor. From the following text return ONLY JSON containing keys: name, summary, skills, education, projects, experience, achievements, certifications, email, phone, github, linkedin, location.",
    "resume_summary_prompt": "Summarize this resume in 5 lines:",
    "project_summary_prompt": "Summarize this project in 3 lines and give 3 keywords:",
    "rag_fallback_intro": "Use the context below (do not invent facts). If the answer is not present, say you don't know and give helpful next steps.",
    "max_context_chars": 3000
}
"""
# ensure prompt_config.json exists before first load
if not os.path.exists(PROMPT_CONFIG):
    os.makedirs(os.path.dirname(PROMPT_CONFIG), exist_ok=True)
    with open(PROMPT_CONFIG, "w") as f:
        json.dump({
            "resume_extraction_prompt": "You are a precise resume extractor. From the following text return ONLY JSON containing keys: name, summary, skills, education, projects, experience, achievements, certifications, email, phone, github, linkedin, location.",
            "resume_summary_prompt": "Summarize this resume in 5 lines:",
            "project_summary_prompt": "Summarize this project in 3 lines and give 3 keywords:",
            "rag_fallback_intro": "Use the context below (do not invent facts). If the answer is not present, say you don't know and give helpful next steps.",
            "max_context_chars": 3000
        }, f, indent=2)

if not os.path.exists(PROMPT_CONFIG):
    with open(PROMPT_CONFIG, "w") as f:
        json.dump(default_prompt_cfg, f, indent=2)

# Load Gemini client lazily
GEN_API_KEY = os.getenv("GEN_API_KEY", "AIzaSyB-CA-KxRKm33aZBoFsQnc84DDpcnl6lac")
if genai is not None and GEN_API_KEY:
    client = genai.Client(api_key=GEN_API_KEY)
else:
    client = None

def load_prompt_config():
    with open(PROMPT_CONFIG, "r") as f:
        return json.load(f)

def save_prompt_config(cfg):
    with open(PROMPT_CONFIG, "w") as f:
        json.dump(cfg, f, indent=2)

def read_json_safe(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def write_json_safe(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

# simple url extractors
URL_RE = re.compile(r"(https?://[^\s]+)")
GITHUB_RE = re.compile(r"(https?://(www\.)?github\.com/[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+)?)", re.I)
LINKEDIN_RE = re.compile(r"(https?://(www\.)?linkedin\.com/[^\s]+)", re.I)
EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")

def extract_links(text):
    links = re.findall(URL_RE, text)
    links = [l[0] if isinstance(l, tuple) else l for l in links]
    github = next((m[0] for m in GITHUB_RE.findall(text)), "")
    linkedin = next((m[0] for m in LINKEDIN_RE.findall(text)), "")
    emails = EMAIL_RE.findall(text)
    return {"links": links, "github": github, "linkedin": linkedin, "emails": emails}

# ---------- Gemini helper (RAG + normal) ----------
def call_gemini(prompt_text, model="models/gemini-2.5-flash"):
    """
    Minimal wrapper that calls Gemini. If client is not configured, return a polite error string.
    """
    if client is None:
        return "Gemini client not configured. Set GEN_API_KEY in your .env."
    try:
        # Use generate_content as you had before
        response = client.models.generate_content(model=model, contents=prompt_text)
        # response may expose .text or nested content
        if hasattr(response, "text"):
            return response.text
        if getattr(response, "candidates", None):
            # try to join candidate texts
            return "\n".join(c[0].content for c in response.candidates)
        return str(response)
    except Exception as e:
        print("Gemini call failed:", e)
        return f"Gemini error: {e}"

def build_rag_context(query, top_k=4):
    """
    Collect top-K snippets from projects and resumes using simple keyword/fuzzy matching.
    This is intentionally lightweight (no heavy embedding library).
    """
    cfg = load_prompt_config()
    projects = read_json_safe(PROJECT_FILE)
    resumes = read_json_safe(RESUME_FILE)

    # flatten into list of (id, type, text)
    candidates = []
    for pid, p in (projects.items() if isinstance(projects, dict) else enumerate(projects)):
        title = p.get("title", "")
        desc = p.get("description", "")
        tech = ", ".join(p.get("technologies", []))
        snippet = f"Project: {title}\nDescription: {desc}\nTechnologies: {tech}\nLinks: {','.join(p.get('links',[]))}"
        candidates.append((pid, "project", snippet))

    for name, r in resumes.items():
        def safe_join_list(lst):
            """Join items that may be strings or dicts."""
            if not lst:
                return ""
            result = []
            for x in lst:
                if isinstance(x, dict):
                    # convert dict to a compact string, e.g. {"company": "X", "role": "Y"} → "company: X, role: Y"
                    result.append(", ".join(f"{k}: {v}" for k, v in x.items()))
                else:
                    result.append(str(x))
            return "; ".join(result)

        snippet = (
            f"Resume: {r.get('name','')}\n"
            f"Summary: {r.get('summary','')}\n"
            f"Skills: {safe_join_list(r.get('skills', []))}\n"
            f"Experience: {safe_join_list(r.get('experience', []))}"
        )

        candidates.append((name, "resume", snippet))

    # Score by keyword overlap
    q_words = set(re.findall(r"\w+", query.lower()))
    scored = []
    for id_, typ, txt in candidates:
        words = set(re.findall(r"\w+", txt.lower()))
        score = len(q_words & words)
        scored.append((score, txt))

    scored.sort(key=lambda x: x[0], reverse=True)
    # take top_k non-zero scores first; if none, take top_k general items
    top_snips = [s for _, s in scored if s][:top_k]
    # join while respecting max_context_chars
    maxc = cfg.get("max_context_chars", 3000)
    context = "\n\n---\n\n".join(top_snips)
    if len(context) > maxc:
        context = context[:maxc]
    return context

# ---------- Routes ----------
@app.route("/")
def index():
    projects = read_json_safe(PROJECT_FILE)
    return render_template("index.html", projects=projects)

@app.route("/add_project", methods=["GET", "POST"])
def add_project():
    if request.method == "POST":
        title = request.form["title"]
        desc = request.form["description"]
        tech = request.form["technologies"]
        importance = request.form["importance"]
        link = request.form.get("link", "").strip()
        file = request.files.get("file")

        filename = None
        if file and file.filename:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))

        # extract links from description and link field
        linkinfo = extract_links(desc + "\n" + (link or ""))
        links = linkinfo["links"]
        github = linkinfo["github"] or (link if "github" in link.lower() else "")
        linkedin = linkinfo["linkedin"] or (link if "linkedin" in link.lower() else "")

        project_id = title.lower().replace(" ", "_")
        project_data = {
            "title": title,
            "description": desc,
            "technologies": [t.strip() for t in tech.split(",")] if tech else [],
            "importance": importance,
            "file": filename,
            "link": link,
            "links": links,
            "github": github,
            "linkedin": linkedin,
            # command is used to run local file if present
            "run_command": f"python3 {os.path.join('projects', filename)}" if filename else "",
            "keywords": [k.strip().lower() for k in (request.form.get("keywords") or "").split(",") if k.strip()]
        }

        projects = read_json_safe(PROJECT_FILE)
        if not isinstance(projects, dict):
            projects = {}
        projects[project_id] = project_data
        write_json_safe(PROJECT_FILE, projects)
        return redirect(url_for("index"))
    return render_template("add_project.html")

@app.route("/upload_resume", methods=["POST"])
def upload_resume():
    if "resume" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["resume"]
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    text = ""
    if filename.lower().endswith(".pdf"):
        with open(filepath, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for p in reader.pages:
                t = p.extract_text()
                if t:
                    text += t + "\n"
    elif filename.lower().endswith((".docx", ".doc")):
        text = docx2txt.process(filepath)
    else:
        return jsonify({"error": "Unsupported file type"}), 400

    # call Gemini to extract JSON (if available), else use simple heuristics
    cfg = load_prompt_config()
    prompt = cfg.get("resume_extraction_prompt") + "\n\nResume text:\n" + text
    extracted = {}
    if client is not None:
        resp = call_gemini(prompt)
        # try to get JSON inside response
        m = re.search(r"(\{.*\})", resp, re.DOTALL)
        if m:
            try:
                extracted = json.loads(m.group(1))
            except Exception:
                extracted = {}
        else:
            extracted = {}
    # fallback simple heuristics
    if not extracted:
        links = extract_links(text)
        extracted = {
            "name": os.path.splitext(filename)[0],
            "summary": call_gemini(cfg.get("resume_summary_prompt") + "\n\n" + text) if client else text[:500],
            "skills": re.findall(r"(?:Skills|TECHNICAL SKILLS|SKILLS)[:\n]*(.*)", text, re.IGNORECASE)[:1] or [],
            "education": [],
            "projects": [],
            "experience": [],
            "achievements": [],
            "certifications": [],
            "email": links.get("emails")[0] if links.get("emails") else "",
            "phone": "",
            "github": links.get("github", ""),
            "linkedin": links.get("linkedin", ""),
            "location": ""
        }

    # normalize extracted items
    name = extracted.get("name") or os.path.splitext(filename)[0]

    # --- NEW: normalize GitHub & LinkedIn ---
    for key in ["github", "linkedin"]:
        val = extracted.get(key, "")
        if val:
            val = val.strip()
            # prepend https:// if missing
            if not val.startswith("http"):
                val = "https://" + val
            extracted[key] = val

    # --- NEW: normalize location ---
    if not extracted.get("location"):
        # simple heuristic: extract from text if missing
        m = re.search(r"(Andhra Pradesh|Hyderabad|India|Chennai|Bangalore|Delhi)", text, re.I)
        if m:
            extracted["location"] = m.group(0)

    resume_store = read_json_safe(RESUME_FILE)

    resume_store[name] = {
        "name": name,
        "summary": extracted.get("summary", "") or "",
        "skills": extracted.get("skills", []) or [],
        "education": extracted.get("education", []) or [],
        "projects": extracted.get("projects", []) or [],
        "experience": extracted.get("experience", []) or [],
        "achievements": extracted.get("achievements", []) or [],
        "certifications": extracted.get("certifications", []) or [],
        "email": extracted.get("email", ""),
        "phone": extracted.get("phone", ""),
        "github": extracted.get("github", ""),
        "linkedin": extracted.get("linkedin", ""),
        "location": extracted.get("location", ""),
        "full_text": text
    }
    write_json_safe(RESUME_FILE, resume_store)

    return jsonify({"name": name, "summary": resume_store[name]["summary"]})

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json or {}
    query_raw = (data.get("query", "") or "").strip()
    query = query_raw.lower()
        # ---------------------------------------------
    # GENERAL GPT AGENT (SECOND AGENT)
    # ---------------------------------------------
    general_agent = data.get("general_agent", False)

    if general_agent:
        # direct LLM mode — skip resume/project logic completely
        prompt = f"General question from user:\n\n{query_raw}\n\nAnswer clearly and accurately:"
        llm_output = call_gemini(prompt)
        return jsonify({"response": llm_output})
    # ---------------------------------------------

    response = "Sorry, I don’t know about that yet."
    found_project = None
    target_resume = None

    resumes = read_json_safe(RESUME_FILE)
    projects = read_json_safe(PROJECT_FILE)

    # 1) Try direct resume match by name or keywords
    if resumes:
        # match by name substring
        for rname, rinfo in resumes.items():
            if rname.lower() in query:
                target_resume = rinfo
                break
        # fuzzy fallback: try close name matches
        if not target_resume:
            names = list(resumes.keys())
            matches = get_close_matches(query_raw, names, n=1, cutoff=0.6)
            if matches:
                target_resume = resumes[matches[0]]

    # helper to present lists
    def join_list(lst):
        if not lst:
            return ""
        if isinstance(lst, list):
            return ", ".join(str(x) for x in lst)
        return str(lst)

    resume_keywords = [
        "skill", "skills", "education", "project", "projects",
        "experience", "achievement", "achievements",
        "certification", "certifications",
        "email", "phone", "contact",
        "location", "address",
        "linkedin", "github",
        "resume", "details", "about", "summary"
    ]

    # 2) If resume intent, answer from resume
    if target_resume and any(k in query for k in resume_keywords):
        if "skill" in query or "skills" in query:
            response = join_list(target_resume.get("skills", [])) or "No skills listed."
        elif "education" in query:
            response = "\n".join(target_resume.get("education", [])) or "No education info."
        elif "project" in query or "projects" in query:
            response = "\n\n".join(str(p) for p in target_resume.get("projects", [])) or "No projects found."
        elif "experience" in query:
            response = "\n\n".join(str(x) for x in target_resume.get("experience", [])) or "No experience found."
        elif "achievement" in query or "achievements" in query:
            response = join_list(target_resume.get("achievements", [])) or "No achievements."
        elif "certification" in query:
            response = join_list(target_resume.get("certifications", [])) or "No certifications."
        elif "email" in query:
            response = target_resume.get("email", "No email found.")
        elif "phone" in query or "contact" in query:
            response = target_resume.get("phone", "No contact found.")
        elif "linkedin" in query:
            link = target_resume.get("linkedin", "")
            if any(k in query for k in ["open", "go to", "execute", "run"]):
                if open_in_browser(link):
                    response = f"Opened LinkedIn profile: {link}"
                else:
                    response = "Couldn't open LinkedIn link."
            else:
                response = f"LinkedIn: {link}" if link else "No LinkedIn link found."


        elif "github" in query:
            link = target_resume.get("github", "")
            if any(k in query for k in ["open", "go to", "execute", "run"]):
                if open_in_browser(link):
                    response = f"Opened GitHub profile: {link}"
                else:
                    response = "Couldn't open GitHub link."
            else:
                response = f"GitHub: {link}" if link else "No GitHub link found."

       
        # --- NEW: combined LinkedIn + GitHub multi-link handler ---
        elif all(k in query for k in ["linkedin", "github"]):
            linkedin = target_resume.get("linkedin", "")
            github = target_resume.get("github", "")
            links_to_open = [l for l in [linkedin, github] if l]
            if any(k in query for k in ["execute", "open", "run", "go to"]):
                if links_to_open:
                    open_in_browser(links_to_open)
                    response = "Opened the following links:\n" + "\n".join(links_to_open)
                else:
                    response = "No LinkedIn or GitHub links found to open."
            else:
                response = f"LinkedIn: {linkedin or 'Not found'}\nGitHub: {github or 'Not found'}"
        else:
            response = target_resume.get("summary", "No summary available.")
       
    else:
        # 3) Try project matching by title / keywords
        # projects might be dict
        items = projects.items() if isinstance(projects, dict) else enumerate(projects)
        for pid, proj in items:
            title = proj.get("title", "").lower()
            keywords = [k.lower() for k in proj.get("keywords", [])] if proj.get("keywords") else []
            if title and title in query or any(k in query for k in keywords) or any(k in query for k in proj.get("technologies", [])):
                found_project = proj
                break

        if found_project:
            # handle run/open semantics
            if any(k in query for k in ["run", "execute", "start", "open", "go to"]):
                # prefer explicit run command (local file) else link else general description
                if any(x in query for x in ["github", "link", "open"]):
                    # open project link if present
                    link = found_project.get("link") or found_project.get("github") or (found_project.get("links")[0] if found_project.get("links") else "")
                    if link:
                        webbrowser.open(link)
                        response = f"Opening {link}..."
                    elif found_project.get("run_command"):
                        response = f"Run command: {found_project.get('run_command')}"
                    else:
                        response = "No link or run command available."
                else:
                    if found_project.get("run_command"):
                        response = f"Run command: {found_project.get('run_command')}"
                    elif found_project.get("link"):
                        webbrowser.open(found_project["link"])
                        response = f"Opening {found_project['link']}..."
                    else:
                        response = "No file or link available to run."
            elif any(w in query for w in ["what", "about", "describe", "description", "info", "details"]):
                response = found_project.get("description", "No description.")
            elif any(w in query for w in ["tech", "technology", "used", "stack"]):
                response = ", ".join(found_project.get("technologies", [])) or "No tech info."
            elif any(k in query for k in ["open project link", "project link", "execute project link", "run project link"]):
                link = (
                    found_project.get("link")
                    or found_project.get("github")
                    or (found_project.get("links")[0] if found_project.get("links") else "")
                )
                if open_in_browser(link):
                    response = f"Opened project link: {link}"
                else:
                    response = "No valid project link found."

            else:
                response = f"{found_project.get('title')} — {found_project.get('description','')}"
        # --- NEW: execute all projects ---
        elif any(k in query for k in ["execute projects", "run all projects", "open projects"]):
            projects = read_json_safe(PROJECT_FILE)
            opened_links = []
            for proj_id, proj in projects.items():
                link = proj.get("link") or proj.get("github") or (proj.get("links")[0] if proj.get("links") else "")
                if link:
                    open_in_browser(link)
                    opened_links.append(link)
            if opened_links:
                response = "Opened all project links:\n" + "\n".join(opened_links)
            else:
                response = "No project links found to execute."

        else:
            # 4) No local direct match -> create RAG context and call Gemini
            context = build_rag_context(query, top_k=5)
            cfg = load_prompt_config()
            rag_intro = cfg.get("rag_fallback_intro", "")
            prompt = f"{rag_intro}\n\nContext:\n{context}\n\nUser query:\n{query_raw}\n\nAnswer:"
            gemini_ans = call_gemini(prompt)
            response = gemini_ans or "Sorry, I couldn't get an answer from Gemini."

    # Save to chat history
    if not os.path.exists(CHAT_HISTORY_FILE):
        write_json_safe(CHAT_HISTORY_FILE, {})

    chat_history = read_json_safe(CHAT_HISTORY_FILE)
    project_name = found_project.get("title") if found_project else ("Resume" if target_resume else "General")
    chat_history.setdefault(project_name, []).append({"user": query_raw, "agent": response})
    write_json_safe(CHAT_HISTORY_FILE, chat_history)

    return jsonify({"response": response})

@app.route("/dashboard")
def dashboard():
    projects_data = read_json_safe(PROJECT_FILE)
    items = projects_data.values() if isinstance(projects_data, dict) else projects_data
    tech_list = []
    importance_list = []
    for p in items:
        tech_list.extend(p.get("technologies", []))
        importance = p.get("importance", "").strip()
        if importance:
            importance_list.append(importance)
    tech_count = dict(Counter(tech_list))
    importance_count = dict(Counter(importance_list))
    return render_template("dashboard.html", tech_count=tech_count, importance_count=importance_count)
    
@app.route("/fix_duplicate_resumes")
def fix_duplicate_resumes():
    resumes = read_json_safe(RESUME_FILE)
    merged = {}
    for name, data in resumes.items():
        norm = re.sub(r"[^A-Za-z0-9 ]+", "", name).strip().title()
        if norm in merged:
            # merge missing fields
            for k, v in data.items():
                if not merged[norm].get(k):
                    merged[norm][k] = v
        else:
            merged[norm] = data
    write_json_safe(RESUME_FILE, merged)
    return jsonify({"message": f"Merged and normalized {len(merged)} resumes."})

# --- simple config endpoints for prompts ---
@app.route("/config", methods=["GET", "POST"])
def config():
    if request.method == "POST":
        cfg = request.json or {}
        cur = load_prompt_config()
        cur.update(cfg)
        save_prompt_config(cur)
        return jsonify({"ok": True, "config": cur})
    else:
        return jsonify(load_prompt_config())

if __name__ == "__main__":
    app.run(debug=True)

