# Job Application Assistant Agent
An LLM-based agent (LangChain / LangGraph + OpenRouter) with a simple **web UI**. Upload **any resume** (PDF / DOCX / TXT) and a
**job description** (upload or paste); the agent compares them with tools, finds missing skills, computes a **Resume-JD Skill Match Score**,
writes an improvement report, and answers follow-up questions from memory.
Built for AI IA-II, Activity 1 (Agentic AI). No model is trained; the focus is reasoning, tool use and orchestration.

## 1. Prerequisites
- Python 3.10+ (tested on 3.12)
- A free OpenRouter API key: https://openrouter.ai/keys (not needed for `--offline` testing)

## 2. Installation
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Environment variables
```bash
cp .env.example .env               # Windows: copy .env.example .env
```
| Variable | Meaning |
|---|---|
| `OPENROUTER_API_KEY` | your key (never commit `.env`) |
| `OPENROUTER_MODEL` | default `openrouter/free`. If tool calling is unreliable with it, pin a free model that **supports tool calling** (https://openrouter.ai/models?q=free). |

## 4. Run
```bash
python app.py                      # web UI at http://127.0.0.1:8000  (real LLM)
python app.py --port 5000          # another port
python tests/run_tests.py          # automated checks (offline stand-in model, no API key)
```
CLI (same agent, no UI): `python agent.py --resume my_cv.docx --jd job.txt --ask "What are my three biggest skill gaps?"`

## 5. Using the UI
1. **Upload Resume** (`.pdf`, `.docx`, `.txt`, `.md`, max 5 MB).
2. **Job Description**: upload a file **or** paste the text (not both).
3. Click **Analyze Resume** (a minute or so with a free model). *Try with sample data* runs the demo files in `data/`.
4. You get: match score, matched / missing skills, experience analysis, major gaps, resume recommendations, learning areas,
   the **agent execution trace** (goal -> tool calls -> observations -> final response), and **Download report**.
5. Ask follow-ups such as *"What are my three biggest skill gaps?"* - the agent answers from its conversation memory without re-reading files.

Uploaded files are saved to a temporary folder only for the run and deleted afterwards. The text is sent to the LLM provider.

## 6. Architecture
```
Web UI (app.py: uploads / pasted JD) -> LLM Agent (agent.py) <-> Memory (MemorySaver + SESSION working memory)
                  | tool call / observation loop (trace in agent_trace.py)
   load_documents -> analyze_skill_gap -> calculate_match_score -> generate_report -> final answer + report
```
See `diagrams/architecture_diagram.png`.
Files: `app.py` (UI) | `agent.py` (agent, memory, CLI, `AnalysisSession`) | `tools.py` (4 tools) | `agent_trace.py` (execution trace) |
`offline_llm.py` (test stand-in only) | `tests/run_tests.py` | `report/` (PDF report + generator) | `data/` (demo and test inputs).

## 7. Tools
| Tool | Purpose |
|---|---|
| `load_documents(resume_path, jd_path)` | reads .txt/.md/.pdf/.docx once into working memory (paths come from the user's upload) |
| `analyze_skill_gap()` | required / matched / missing skills, preferred skills, relevant experience |
| `calculate_match_score()` | matched required skills / required skills x 100 |
| `generate_report(recommendations, learning_areas, summary)` | structured markdown report; the LLM supplies the advice text |

Skills are matched against a lexicon in `tools.py` (`SKILLS`, ~35 skills); add entries for other job roles.
The score is a keyword-overlap indicator, **not** a hiring probability.

## 8. How this demonstrates Agentic AI
- **Goal, not a script:** the UI sends a goal with the uploaded file paths; the LLM chooses which tool to call next from the goal and observations.
- **Multi-step tool use:** four chained tool calls; a tool error changes what the agent does next (it explains instead of continuing).
- **Memory:** `MemorySaver` keeps the conversation (one thread per browser session); `SESSION` lets tools share parsed documents.
- **Observable behaviour:** `agent_trace.py` records actions and observations (never hidden reasoning) to `logs/agent_trace.txt` and to the UI.

## 9. Tests
`python tests/run_tests.py` covers: normal analysis, follow-up memory, different resumes/JDs (output must change), TXT/PDF/DOCX uploads,
pasted JD, missing/empty/corrupt/unsupported inputs, tool failure, report generation and download, missing API key, upload cleanup, and a
check that the tools never reference the sample files. It uses the offline stand-in, so it checks the plumbing, **not** the quality of a real LLM.

## Limitations
Keyword-based skill detection (synonyms outside the lexicon are missed); scanned PDFs are not supported; free LLMs vary in tool-calling quality
and rate limits and their wording differs between runs; the UI is single-user/local (analyses run one at a time).
`offline_llm.py` is not an LLM - use the live mode for the demo.
