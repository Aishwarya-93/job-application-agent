"""Builds report/Job_Application_Agent_Report.pdf  (run from the project root: python report/make_report.py)

Optional: python report/make_report.py --members "A (Roll 1), B (Roll 2)" --batch "B2"
"""
import argparse
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle, PageBreak

_ap = argparse.ArgumentParser()
_ap.add_argument("--members", default="[Name 1, Roll No.]  /  [Name 2, Roll No.]  /  [Name 3, Roll No.]")
_ap.add_argument("--batch", default="[Batch]")
ARGS = _ap.parse_args()

ss = getSampleStyleSheet()
NAVY = colors.HexColor("#1f3b73")
B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=10, leading=14, spaceAfter=5)
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=14, spaceBefore=10, spaceAfter=5, textColor=NAVY)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=11, spaceBefore=6, spaceAfter=3, textColor=NAVY)
CODE = ParagraphStyle("C", fontName="Courier", fontSize=6.9, leading=8.6, backColor=colors.HexColor("#f4f4f4"), borderPadding=4, spaceAfter=8)
NOTE = ParagraphStyle("N", parent=B, backColor=colors.HexColor("#fff3c4"), borderPadding=5, fontSize=9, spaceBefore=3, spaceAfter=8)
TC = ParagraphStyle("TC", parent=B, fontSize=8.8, leading=11, spaceAfter=0)
TH = ParagraphStyle("TH", parent=TC, textColor=colors.white, fontName="Helvetica-Bold")

def tbl(rows, widths, header=True):
    data = [[Paragraph(str(c), TH if header and i == 0 else TC) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header: st.append(("BACKGROUND", (0, 0), (-1, 0), NAVY))
    t.setStyle(TableStyle(st)); return t

def shot(text, h=3.6):
    t = Table([[Paragraph(text, ParagraphStyle("p", parent=B, alignment=1, textColor=colors.grey))]], colWidths=[16*cm], rowHeights=[h*cm])
    t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 1, colors.grey, None, (4, 2)), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")])); return t

# ---- real artifacts produced by the code
import textwrap
trace = open("logs/example_trace.txt", encoding="utf-8").read().split("\nFinal response to user:\n")
out = [trace[0]]
for block in trace[1:]:
    body, _, rest = block.partition("\n\n(tools used")
    lines = body.splitlines()
    out.append("\nFinal response to user:")
    if len(lines) > 4:
        out += [lines[0], "  [full report text omitted here - see outputs/example_report.md]"]
    else:
        out += lines
    out.append("\n(tools used" + rest)
wrapped = []
for l in "\n".join(out).splitlines():
    wrapped += textwrap.wrap(l, 118, subsequent_indent="                   ", replace_whitespace=False, drop_whitespace=False) or [""]
trace_txt = "\n".join(wrapped)
report_md = open("outputs/example_report.md", encoding="utf-8").read()
test_rows = [["Result", "Test scenario"]] + [l.split("  ", 1) for l in open("logs/test_results.txt").read().splitlines()[:-1]]
test_sum = open("logs/test_results.txt").read().splitlines()[-1]

S = [Paragraph("Agentic AI: Job Application Assistant", ParagraphStyle("T", parent=ss["Title"], fontSize=21)),
     Paragraph("AI - IA II, Activity 1: Agentic AI System using Modern LLM-based Tools", ParagraphStyle("s", parent=B, alignment=1, fontSize=11.5)),
     Spacer(1, 6),
     tbl([["Group members", ARGS.members], ["Lab batch", ARGS.batch],
          ["Course / Faculty", "Artificial Intelligence / Dr. Ayesha Hakim"],
          ["Stack", "Python, LangChain / LangGraph, OpenRouter (LLM API), Flask (web UI), pypdf, python-docx"]], [4*cm, 12*cm], header=False),
     Spacer(1, 8)]

S += [Paragraph("1. Introduction", H1),
      Paragraph("A classical AI agent perceives its environment, decides, and acts using rules written by a programmer. <b>Agentic AI</b> keeps this "
                "perceive-decide-act idea but lets a large language model (LLM) make the decisions: the LLM receives a <i>goal</i>, plans steps, calls "
                "<i>tools</i>, reads the results and repeats until it can give a final answer. This project builds a small agent of this kind. "
                "No model is trained; the work is in tool design, orchestration, memory and an observable execution trace.", B),
      Paragraph("2. Problem Statement", H1),
      Paragraph("Students applying for jobs rarely know how well their resume covers what a job description (JD) asks for. The task is to build an agent that "
                "analyses <i>any</i> resume uploaded by the user against <i>any</i> JD (uploaded or pasted), finds skill gaps, computes a simple match score and recommends honest improvements, "
                "through a simple web page.", B),
      Paragraph("3. Agent Goal", H1),
      Paragraph("<b>Goal:</b> given a user-uploaded resume (PDF / DOCX / TXT) and a JD (file or pasted text), produce (a) a Resume-JD Skill Match Score, (b) matched and missing skills, (c) relevant experience, "
                "(d) the major gaps, (e) concrete resume improvements and learning areas, and (f) a saved report. The user can then ask follow-up questions.", B),
      Paragraph("4. Agent Role", H1),
      Paragraph("A single <b>LLM orchestrator</b> (an OpenRouter model used through LangChain, driven by a LangGraph ReAct-style loop). It does not follow a fixed script: "
                "from the goal, the system prompt, the tool descriptions and the observations so far it decides <i>which tool to call next</i>, "
                "decides when enough information is available, writes the recommendations, and answers follow-ups. Its limits are fixed by the system prompt "
                "(use tool outputs as facts, never invent experience, no hiring predictions).", B),
      Paragraph("5. Tools", H1),
      tbl([["Tool", "What it does", "Why it is needed"],
           ["load_documents", "Reads the user-uploaded resume and JD (.pdf / .docx / .txt) from the paths given in the goal, once, and stores them in working memory.", "The LLM cannot open files; also avoids passing long text through the prompt."],
           ["analyze_skill_gap", "Matches resume against JD skills (lexicon, word-boundary regex): required, matched, missing, preferred, relevant experience lines.", "Deterministic, repeatable skill detection; the LLM reasons over its result."],
           ["calculate_match_score", "Score = matched required skills / required skills x 100, plus a band (limited / moderate / strong overlap).", "Arithmetic is delegated to code - LLMs can miscalculate."],
           ["generate_report", "Builds the structured markdown report from working memory and the advice written by the LLM; saves it.", "Gives a consistent, complete deliverable."]],
          [3.3*cm, 6.9*cm, 5.8*cm]),
      Paragraph("All tools return JSON with a status and a short summary, and never raise exceptions: failures come back as error observations the agent can explain. "
                "No tool reads a fixed file: the file paths come from the goal message, which the web UI builds from the user's uploads.", B),
      Paragraph("6. Memory", H1),
      Paragraph("<b>Conversation memory (short-term / contextual):</b> a LangGraph <font face='Courier'>MemorySaver</font> checkpointer stores every message, tool call and observation under a "
                "thread id. When the user later asks 'What are my three biggest skill gaps?', the LLM sees the earlier analysis and answers without reloading files. "
                "<b>Working memory:</b> a small <font face='Courier'>SESSION</font> dictionary shared by the tools holds the parsed documents, the gap analysis and the score, "
                "so each file is read once per analysis. In the web UI each browser session has its own agent and conversation thread, so follow-up questions refer to that user's own resume and JD. "
                "Memory lasts for the running process only; no database is used.", B)]

S += [Paragraph("7. Planning Strategy", H1),
      Paragraph("The agent uses a <b>ReAct-style</b> loop (Reason, Act, Observe). The system prompt describes the tools and the working rules but does <i>not</i> hard-code a call order; "
                "tool descriptions state prerequisites (for example, the score tool needs the gap analysis first). In a normal run the LLM chooses "
                "load, analyse, score, report. If a tool reports an error, the agent explains the problem instead of continuing; on follow-ups it answers from memory without any tool call.", B),
      Paragraph("8. System Architecture", H1),
      Image("diagrams/architecture_diagram.png", width=15*cm, height=15*cm*0.64),
      Paragraph("<i>Figure 1: LLM orchestrator, four tools, conversation and working memory, trace logger.</i>", B),
      Paragraph("9. Agent Workflow", H1),
      Paragraph("<b>Goal</b> (user uploads resume + JD and clicks Analyze; the UI turns this into the goal message) &rarr; <b>Planning</b> (LLM decides it needs the documents first) &rarr; <b>Tool call</b> (load_documents) &rarr; "
                "<b>Observation</b> (documents loaded) &rarr; <b>Next action</b> (analyze_skill_gap, then calculate_match_score, then generate_report) &rarr; "
                "<b>Final output</b> (score, skills, gaps, advice, saved report). Steps 3-5 repeat until the goal is complete.", B),
      Paragraph("10. Implementation", H1),
      tbl([["File", "Role"],
           ["app.py", "Flask web UI: upload resume (PDF/DOCX/TXT), upload or paste JD, Analyze button, results, execution trace, report download, follow-up chat. Saves uploads to a temporary folder, passes the paths to the agent, deletes them afterwards."],
           ["agent.py", "Builds the OpenRouter LLM and the agent (create_agent + MemorySaver), AnalysisSession (one agent + memory thread per browser session), error handling, CLI (--resume, --jd, --ask, --offline)."],
           ["tools.py", "The four tools, the skill lexicon, safe_tool wrapper (turns exceptions into error observations), SESSION working memory."],
           ["agent_trace.py", "LangChain callback handler: writes the execution trace to logs/ and keeps it as structured events that the UI displays."],
           ["offline_llm.py", "Rule-based stand-in for the LLM, used only for offline testing."],
           ["tests/run_tests.py", "Automated checks for tools, agent, memory, report, uploads and the web UI (Section 13)."],
           ["data/, logs/, outputs/, diagrams/, report/", "Demo/test inputs (the samples are used only by the demo button, the CLI default and tests), traces, generated reports, diagram, this report."]], [4.2*cm, 11.8*cm])]

S += [Paragraph("11. Agent Interaction Logs", H1),
      Paragraph("The trace records the goal, each step's tool selected, tool input and observation, and the final response. It deliberately logs <i>actions and observations</i>, not the model's hidden reasoning.", B),
      Paragraph("<b>About this trace:</b> it was produced for a <i>non-sample</i> resume and JD (a web-developer resume, data/example_resume_webdev.txt, against a Frontend Developer JD) "
                "with the <b>--offline</b> rule-based stand-in model, because the build environment had no OpenRouter access. A live run writes the same format to logs/agent_trace.txt, "
                "and the web UI shows the same steps as a checklist (Goal received, Documents loaded, Skill gaps analyzed, Match score calculated, Report generated, Final response prepared) "
                "with the tool name of each step. To refresh this section with a live trace, replace logs/example_trace.txt and run report/make_report.py.", B),
      Preformatted(trace_txt, CODE)]

S += [Paragraph("12. Web UI, Sample Inputs and Outputs", H1),
      Paragraph("<b>UI.</b> The page (python app.py) has: Upload Resume (PDF / DOCX / TXT); Job Description as an uploaded file <i>or</i> pasted text; an <b>Analyze Resume</b> button; "
                "and a 'Try with sample data' button for demos. After the agent finishes, the page shows the match score, matched and missing skills, experience analysis, major gaps, "
                "resume improvement recommendations, suggested learning areas, a summary, the agent execution trace and a Download report button, followed by a follow-up question box "
                "that uses the same agent memory.", B),
      Paragraph("<b>Input (example):</b> data/example_resume_webdev.txt (web developer: JavaScript, React, HTML, CSS, Node.js, Git, REST APIs) and data/example_jd_frontend.txt "
                "(Frontend Developer, 8 required and 2 preferred skills). <b>Output</b> (generated report, abridged):", B),
      Preformatted("\n".join(l for l in report_md.splitlines() if l.strip())[:1900], CODE),
      Paragraph("Required skills: 8; matched: 6; missing: 2 (Docker, AWS). Skill Match Score = 6/8 = <b>75%</b>. A different resume or JD gives a different result "
                "(for example, a web resume against a DevOps JD gives 0%); this is covered by the tests. "
                "The score is a keyword-overlap indicator, not a hiring probability; preferred skills are reported separately and are not part of the score.", B)]

S += [Paragraph("13. Testing", H1),
      Paragraph("Run with <font face='Courier'>python tests/run_tests.py</font> using the offline stand-in model. This verifies the tools, memory, tracing, error handling, file uploads (TXT / PDF / DOCX), "
                "pasted JDs, missing-input handling and the web UI routes; it does <b>not</b> measure the reasoning quality of a real LLM.", B),
      tbl(test_rows, [1.6*cm, 14.4*cm]), Paragraph(f"<b>{test_sum}</b>", B),
      Paragraph("The most important check is that the system is not tied to the sample files: tests 12 and 12b analyse different resume/JD pairs and require different scores and missing-skill lists, "
                "test 11 requires the downloaded report to contain the uploaded resume's text, and test 16 checks that the tools never reference the sample files.", B)]

S += [Paragraph("14. Classical vs Modern Agent Comparison", H1),
      tbl([["Classical Agent (Syllabus)", "Modern Agent (This Activity)", "How it appears in this project"],
           ["Rule-based", "LLM-driven", "The LLM decides which tool to call; only the skill matcher is rule-based, inside a tool."],
           ["Static logic", "Dynamic planning", "The call order is not hard-coded; errors and follow-ups change what the agent does next."],
           ["No memory", "Contextual memory", "MemorySaver keeps the conversation; SESSION shares data between tools."],
           ["Single-step", "Multi-step reasoning", "Four chained tool calls, each using the previous observation."]],
          [4.2*cm, 4.2*cm, 7.6*cm]),
      Paragraph("A classical simple-reflex agent for the same task would map percepts to actions with fixed IF-THEN rules (for example, 'IF a required skill word is absent THEN flag it') "
                "and could not take a new goal, remember an earlier result or explain a failure. The modern agent is more flexible but less predictable and needs a trace to be trusted. "
                "This project is a <b>hybrid</b>: reliable rule-based logic sits inside the tools, the LLM orchestrates them.", B),
      Paragraph("15. Analysis and Reflection", H1),
      Paragraph("<b>What worked.</b> Small tools with clear descriptions and prerequisites make orchestration easy to follow. Returning a one-line summary from every tool made the trace readable. "
                "Sharing data through working memory removed redundant file reads and kept long text out of the prompt.", B),
      Paragraph("<b>Tool orchestration and reliability.</b> The agent depends on the LLM choosing correct tool calls; free OpenRouter models differ in tool-calling quality and rate limits, "
                "so a run can fail or vary. Failures are caught and reported (no raw traceback). Calculation is delegated to a tool for reliability.", B),
      Paragraph("<b>Limitations.</b> Skill detection is keyword-based with a lexicon of about 35 skills (synonyms outside it are missed; it cannot judge depth or seniority). The score is a simple ratio, not a real ATS or hiring result. "
                "Scanned PDFs are unsupported. Memory is lost when the program exits. The agent is not fully autonomous: it works with four predefined tools inside limits set by the prompt. "
                "The resume text is sent to the LLM provider through the API. The web UI is a single-user local app: analyses run one at a time and the tools share one working memory. "
                "Uploaded files are deleted after each analysis; the generated report and trace log stay on disk.", B),
      Paragraph("<b>Future improvements.</b> Semantic (embedding) matching of skills, a reviewer agent that checks the recommendations, persistent memory, support for more job roles, multi-user support.", B),
      Paragraph("16. Conclusion", H1),
      Paragraph("The project extends the classical goal-based agent into an agentic system: through a simple web UI, a user uploads any resume and JD; an LLM receives the goal, chooses among four tools, observes the results, keeps context across turns and produces a structured, "
                "goal-oriented output, with every step visible in a trace. It meets the activity requirements of one LLM agent, at least two tools, multi-step reasoning and observable behaviour.", B)]

def footer(c, d):
    c.saveState(); c.setFont("Helvetica", 8); c.setFillColor(colors.grey)
    c.drawCentredString(A4[0]/2, 1.0*cm, f"Agentic AI Job Application Assistant  |  Page {d.page}"); c.restoreState()

SimpleDocTemplate("report/Job_Application_Agent_Report.pdf", pagesize=A4, leftMargin=2.2*cm, rightMargin=2.2*cm,
                  topMargin=1.8*cm, bottomMargin=1.8*cm, title="Agentic AI Job Application Assistant").build(S, onFirstPage=footer, onLaterPages=footer)
print("report ok")
