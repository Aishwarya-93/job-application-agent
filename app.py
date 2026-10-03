"""Web UI for the Job Application Assistant agent.

    python app.py              # real LLM through OpenRouter (needs OPENROUTER_API_KEY in .env)
    python app.py --offline    # development only: rule-based stand-in instead of an LLM

The user uploads ANY resume (.pdf/.docx/.txt) and a job description (upload or paste). The files are saved
to a temporary folder and their paths are handed to the existing LLM agent, which calls its tools
(load_documents -> analyze_skill_gap -> calculate_match_score -> generate_report). Follow-up questions go
to the same agent thread, so MemorySaver gives it the earlier analysis. Only an execution trace
(tool calls + observations) is shown, never the model's private reasoning.
"""
import argparse
import os
import re
import secrets
import shutil
import tempfile
import threading
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template_string, request, send_file, session

import tools
from agent import AnalysisSession, MissingKeyError, build_llm, model_name
from tools import SUPPORTED_EXTENSIONS, read_document

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "outputs" / "improvement_report.md"
SAMPLE_RESUME, SAMPLE_JD = ROOT / "data" / "sample_resume.txt", ROOT / "data" / "sample_jd.txt"
MAX_UPLOAD_MB = 5

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY") or secrets.token_hex(16)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
app.config["OFFLINE"] = False                 # set by --offline (development only)
app.config["LLM_FACTORY"] = None              # tests may inject a factory returning a chat model

SESSIONS: dict = {}                           # browser session id -> AnalysisSession (own memory thread)
RUN_LOCK = threading.Lock()                   # tools share one working memory (tools.SESSION): one run at a time


# ------------------------------------------------------------------ helpers
def _new_llm():
    factory = app.config["LLM_FACTORY"]
    return factory() if factory else build_llm(app.config["OFFLINE"])


def _sid() -> str:
    if "sid" not in session:
        session["sid"] = secrets.token_hex(8)
    return session["sid"]


def _save_upload(file_storage, dest_dir: Path, stem: str, label: str) -> Path:
    """Validate and save one uploaded file. Raises ValueError with a user-readable message."""
    name = file_storage.filename or ""
    ext = Path(name).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"{label}: unsupported file type '{ext or name}'. Use {', '.join(SUPPORTED_EXTENSIONS)}.")
    path = dest_dir / f"{stem}{ext}"          # fixed name: the original file name is never used on disk
    file_storage.save(path)
    return path


def _section(md: str, title: str) -> list:
    m = re.search(rf"^## {re.escape(title)}\s*\n(.*?)(?=^## |\Z)", md, re.S | re.M)
    if not m:
        return []
    return [l[2:].strip() for l in m.group(1).splitlines() if l.startswith("- ")]


def _text_section(md: str, title: str) -> str:
    m = re.search(rf"^## {re.escape(title)}\s*\n(.*?)(?=^## |\Z)", md, re.S | re.M)
    return m.group(1).strip() if m else ""


def _results() -> dict | None:
    """Structured results of the latest analysis, read from the tools' working memory + generated report."""
    gap, score = tools.SESSION.get("gap"), tools.SESSION.get("score")
    if not gap or not score:
        return None
    md = REPORT.read_text(encoding="utf-8") if REPORT.exists() else ""
    return {
        "score": score["score"], "matched_count": score["matched"], "total": score["total"], "band": score["band"],
        "matched_skills": gap["matched_skills"], "missing_skills": gap["missing_skills"],
        "preferred_matched": gap["preferred_matched"], "preferred_missing": gap["preferred_missing"],
        "experience": gap["relevant_experience"], "major_gaps": gap["missing_skills"][:3],
        "recommendations": _section(md, "Recommended Resume Improvements"),
        "learning_areas": _section(md, "Suggested Learning Areas"),
        "summary": _text_section(md, "Overall Summary"),
        "report_ready": bool(md),
    }


def _turn_events(sess: AnalysisSession) -> list:
    t = sess.tracer.turn
    return [e for e in sess.tracer.events if e["turn"] == t]


# ------------------------------------------------------------------ routes
@app.get("/")
def index():
    return render_template_string(PAGE, offline=app.config["OFFLINE"], model=model_name(),
                                  exts=", ".join(SUPPORTED_EXTENSIONS), max_mb=MAX_UPLOAD_MB)


@app.post("/analyze")
def analyze():
    use_sample = request.form.get("use_sample") == "1"
    resume_f, jd_f = request.files.get("resume"), request.files.get("jd_file")
    jd_text = (request.form.get("jd_text") or "").strip()
    has_resume = bool(resume_f and resume_f.filename)
    has_jd_file = bool(jd_f and jd_f.filename)

    # ---- input validation (clear messages, nothing reaches the agent if something is missing)
    if not use_sample:
        if not has_resume:
            return jsonify(ok=False, error="Please upload your resume (PDF, DOCX or TXT)."), 400
        if not has_jd_file and not jd_text:
            return jsonify(ok=False, error="Please upload a job description file or paste the JD text."), 400
        if has_jd_file and jd_text:
            return jsonify(ok=False, error="Provide the job description either as a file OR as pasted text, not both."), 400

    work = Path(tempfile.mkdtemp(prefix="jobassist_"))
    try:
        if use_sample:
            resume_p, jd_p = SAMPLE_RESUME, SAMPLE_JD
        else:
            try:
                resume_p = _save_upload(resume_f, work, "resume", "Resume")
                if has_jd_file:
                    jd_p = _save_upload(jd_f, work, "job_description", "Job description")
                else:
                    jd_p = work / "job_description.txt"
                    jd_p.write_text(jd_text, encoding="utf-8")
                for label, p in (("Resume", resume_p), ("Job description", jd_p)):   # fail early: readable text?
                    read_document(str(p), label)
            except (ValueError, FileNotFoundError) as e:
                return jsonify(ok=False, error=str(e)), 400
            except Exception as e:  # noqa: BLE001  (corrupt PDF/DOCX, etc.)
                return jsonify(ok=False, error=f"Could not read the uploaded file ({type(e).__name__}). "
                                               "Is it a valid, non-scanned document?"), 400

        try:
            llm = _new_llm()
        except MissingKeyError as e:
            return jsonify(ok=False, error=str(e)), 503

        with RUN_LOCK:
            old = SESSIONS.pop(_sid(), None)
            if old:
                old.close()
            tools.SESSION.clear()                  # never show stale results from an earlier run
            REPORT.unlink(missing_ok=True)
            sess = AnalysisSession(llm, header="LLM: " + ("OFFLINE stand-in (not a real LLM)" if app.config["OFFLINE"]
                                                          else model_name()))
            SESSIONS[_sid()] = sess
            answer = sess.analyze(str(resume_p), str(jd_p))
            results = _results()
            events = _turn_events(sess)
        failed = results is None
        return jsonify(ok=not failed, answer=answer, results=results, trace=events,
                       error=("The agent did not complete the analysis. " + answer) if failed else None,
                       offline=app.config["OFFLINE"], sample=use_sample)
    finally:
        shutil.rmtree(work, ignore_errors=True)   # uploaded resumes are not kept on disk


@app.post("/ask")
def ask():
    q = ((request.get_json(silent=True) or {}).get("question") or "").strip()
    if not q:
        return jsonify(ok=False, error="Please type a question."), 400
    sess = SESSIONS.get(_sid())
    if not sess:
        return jsonify(ok=False, error="Run an analysis first, then ask follow-up questions."), 400
    with RUN_LOCK:
        answer = sess.ask(q)
        events = _turn_events(sess)
    return jsonify(ok=True, answer=answer, trace=events)


@app.get("/download")
def download():
    if not REPORT.exists():
        return jsonify(ok=False, error="No report yet. Run an analysis first."), 404
    return send_file(REPORT, as_attachment=True, download_name="resume_improvement_report.md", mimetype="text/markdown")


@app.errorhandler(413)
def too_large(_e):
    return jsonify(ok=False, error=f"File too large (limit {MAX_UPLOAD_MB} MB)."), 413


# ------------------------------------------------------------------ page
PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Job Application Assistant</title>
<style>
 body{font-family:system-ui,Segoe UI,Arial,sans-serif;max-width:900px;margin:24px auto;padding:0 16px;color:#222;background:#fafafa}
 h1{color:#1f3b73;margin-bottom:4px} .sub{color:#666;margin-top:0}
 .card{background:#fff;border:1px solid #ddd;border-radius:8px;padding:16px;margin:14px 0}
 label{font-weight:600;display:block;margin:10px 0 4px} textarea{width:100%;min-height:130px;box-sizing:border-box;font-family:inherit}
 button{background:#1f3b73;color:#fff;border:0;border-radius:6px;padding:10px 18px;font-size:15px;cursor:pointer}
 button.alt{background:#fff;color:#1f3b73;border:1px solid #1f3b73} button:disabled{opacity:.5;cursor:wait}
 .or{color:#888;margin:8px 0;font-weight:600} .err{background:#fdecea;border:1px solid #f5a7a0;color:#8a1c12}
 .warn{background:#fff3c4;border:1px solid #e6cf6e} .score{font-size:44px;font-weight:700;color:#1f3b73}
 .chip{display:inline-block;padding:3px 10px;margin:2px;border-radius:12px;font-size:13px}
 .ok{background:#e1f4e1;color:#1d6b1d} .bad{background:#fde3e1;color:#9a2a20} .pref{background:#eef;color:#335}
 .trace div{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:13px;margin:3px 0} .trace .sub2{margin-left:22px;color:#555}
 .hidden{display:none} pre{white-space:pre-wrap;word-wrap:break-word;font-family:inherit} ul{margin:4px 0;padding-left:22px}
 .chat .q{font-weight:600;margin-top:10px} .small{font-size:13px;color:#666}
</style></head><body>
<h1>AI Job Application Assistant</h1>
<p class="sub">Upload any resume and a job description. An LLM agent analyses them with tools and explains the gaps.</p>
{% if offline %}<div class="card warn"><b>OFFLINE TEST MODE</b> - a rule-based stand-in is used instead of a real LLM. For development only.</div>{% else %}
<p class="small">LLM: <code>{{ model }}</code> via OpenRouter</p>{% endif %}

<div class="card">
 <form id="f">
  <label for="resume">Upload Resume <span class="small">({{ exts }}, max {{ max_mb }} MB)</span></label>
  <input type="file" id="resume" name="resume" accept=".pdf,.docx,.txt,.md">
  <label>Job Description</label>
  <input type="file" id="jd_file" name="jd_file" accept=".pdf,.docx,.txt,.md">
  <div class="or">OR</div>
  <textarea id="jd_text" name="jd_text" placeholder="Paste the job description here"></textarea>
  <p><button type="submit" id="go">Analyze Resume</button>
     <button type="button" class="alt" id="sample">Try with sample data</button></p>
 </form>
 <div id="status" class="small"></div>
</div>
<div id="error" class="card err hidden"></div>

<div id="out" class="hidden">
 <div class="card"><div class="small">Resume-JD Skill Match Score</div><div class="score" id="score"></div><div class="small" id="band"></div></div>
 <div class="card"><b>Matched skills</b><div id="matched"></div><br><b>Missing skills</b><div id="missing"></div>
   <br><b>Preferred skills</b> <span class="small">(not part of the score)</span><div id="pref"></div></div>
 <div class="card"><b>Experience analysis</b><ul id="exp"></ul></div>
 <div class="card"><b>Major gaps</b><ul id="gaps"></ul></div>
 <div class="card"><b>Resume improvement recommendations</b><ul id="recs"></ul></div>
 <div class="card"><b>Suggested learning areas</b><ul id="learn"></ul></div>
 <div class="card"><b>Summary</b><p id="summary"></p></div>
 <div class="card"><b>Agent execution trace</b> <span class="small">(actions and observations only)</span><div class="trace" id="trace"></div></div>
 <p><a id="dl" href="/download"><button type="button">Download report (.md)</button></a></p>
 <div class="card chat"><b>Ask a follow-up</b> <span class="small">- the agent remembers this analysis</span>
  <div id="chat"></div>
  <p><input id="q" style="width:75%" placeholder="What are my three biggest skill gaps?"> <button id="ask" type="button">Ask</button></p>
  <div class="trace" id="ftrace"></div></div>
</div>

<script>
const $ = id => document.getElementById(id);
const li = (ul, items, empty) => { ul.replaceChildren(); (items.length ? items : [empty]).forEach(t => { const e = document.createElement("li"); e.textContent = t; ul.appendChild(e); }); };
const chips = (box, items, cls) => { box.replaceChildren(); if (!items.length) { box.textContent = "none"; return; } items.forEach(t => { const s = document.createElement("span"); s.className = "chip " + cls; s.textContent = t; box.appendChild(s); }); };
function showErr(m) { $("error").textContent = m; $("error").classList.remove("hidden"); }
function drawTrace(box, events) {
  box.replaceChildren();
  events.forEach(e => {
    const d = document.createElement("div");
    if (e.kind === "tool_call") { d.className = "sub2"; d.textContent = "-> tool: " + e.tool; }
    else if (e.kind === "observation") { d.textContent = (e.ok ? "\u2713 " : "\u2717 ") + e.label; const s = document.createElement("div"); s.className = "sub2"; s.textContent = e.detail; box.appendChild(d); box.appendChild(s); return; }
    else d.textContent = (e.ok === false ? "\u2717 " : "\u2713 ") + e.label;
    box.appendChild(d);
  });
}
function show(r, trace) {
  $("score").textContent = r.score + "%"; $("band").textContent = r.matched_count + " of " + r.total + " required skills - " + r.band;
  chips($("matched"), r.matched_skills, "ok"); chips($("missing"), r.missing_skills, "bad");
  chips($("pref"), r.preferred_matched.concat(r.preferred_missing), "pref");
  li($("exp"), r.experience.map(e => e.line + "  (shows: " + e.skills.join(", ") + ")"), "No project/internship lines mention the required skills.");
  li($("gaps"), r.major_gaps, "No major gaps"); li($("recs"), r.recommendations, "(none returned)"); li($("learn"), r.learning_areas, "(none returned)");
  $("summary").textContent = r.summary; drawTrace($("trace"), trace); $("out").classList.remove("hidden");
  $("dl").style.display = r.report_ready ? "" : "none";
}
async function run(fd) {
  $("error").classList.add("hidden"); $("out").classList.add("hidden"); $("chat").replaceChildren(); $("ftrace").replaceChildren();
  $("go").disabled = $("sample").disabled = true; $("status").textContent = "Agent running - this can take a minute...";
  try {
    const res = await fetch("/analyze", {method: "POST", body: fd}); const j = await res.json();
    if (!j.ok) { showErr(j.error || "Analysis failed."); if (j.trace) { $("out").classList.remove("hidden"); drawTrace($("trace"), j.trace); } return; }
    show(j.results, j.trace);
  } catch (e) { showErr("Request failed: " + e); } finally { $("go").disabled = $("sample").disabled = false; $("status").textContent = ""; }
}
$("f").addEventListener("submit", ev => { ev.preventDefault(); run(new FormData($("f"))); });
$("sample").addEventListener("click", () => { const fd = new FormData(); fd.append("use_sample", "1"); run(fd); });
$("ask").addEventListener("click", async () => {
  const q = $("q").value.trim(); if (!q) return; $("ask").disabled = true;
  const box = document.createElement("div"); const qq = document.createElement("div"); qq.className = "q"; qq.textContent = "You: " + q; box.appendChild(qq);
  const a = document.createElement("pre"); a.textContent = "..."; box.appendChild(a); $("chat").appendChild(box); $("q").value = "";
  try { const res = await fetch("/ask", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({question: q})}); const j = await res.json();
    a.textContent = j.ok ? j.answer : j.error; if (j.trace) drawTrace($("ftrace"), j.trace);
  } catch (e) { a.textContent = "Request failed: " + e; } finally { $("ask").disabled = false; }
});
$("q").addEventListener("keydown", e => { if (e.key === "Enter") $("ask").click(); });
</script></body></html>"""


if __name__ == "__main__":
    load_dotenv()
    ap = argparse.ArgumentParser(description="Job Application Assistant web UI")
    ap.add_argument("--offline", action="store_true", help="development only: rule-based stand-in, no API key")
    ap.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    a = ap.parse_args()
    app.config["OFFLINE"] = a.offline
    print(f"Open http://127.0.0.1:{a.port}   (LLM: {'OFFLINE stand-in' if a.offline else model_name()})")
    app.run(host="127.0.0.1", port=a.port, debug=False, threaded=True)
