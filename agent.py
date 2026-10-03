"""Job Application Assistant - an LLM agent built with LangChain/LangGraph + OpenRouter.

The LLM receives a goal and a set of tools. It decides which tool to call next from the goal and
from what it has observed so far, then writes the final answer.

    python app.py                                     # web UI: upload any resume + JD (recommended)
    python agent.py                                   # CLI: sample data, live LLM, then follow-up chat
    python agent.py --resume cv.pdf --jd job.docx     # CLI: your own files
    python agent.py --offline --ask "What are my three biggest skill gaps?"   # no API key needed
"""
import argparse
import os
import sys
import traceback
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError

from agent_trace import TraceHandler
from tools import ALL_TOOLS, SUPPORTED_EXTENSIONS

SYSTEM_PROMPT = """You are a Job Application Assistant agent. Your goal: compare a candidate's resume with a
job description (JD), explain how well they match, and suggest honest improvements.

TOOLS (decide which to call next based on the user's goal and what you have observed):
- load_documents(resume_path, jd_path): reads both files (.txt/.pdf/.docx) into working memory. Needed before analysis.
  Use the exact file paths the user gives you; never use sample or guessed paths.
- analyze_skill_gap(): required / matched / missing skills and relevant experience. Takes no arguments.
- calculate_match_score(): the Resume-JD Skill Match Score. Takes no arguments. Never calculate it yourself.
- generate_report(recommendations, learning_areas, summary): writes the final structured report.

HOW TO WORK
1. Before each tool call, state the next action in one short sentence.
2. Use tool outputs as facts. If a tool returns an error, do not invent results: explain the problem to the user.
3. For FOLLOW-UP questions (e.g. "what are my biggest gaps?") answer from the earlier tool results in this
   conversation. Do not reload files or re-run tools unless the user gives new files.
4. If the user has not given both a resume and a JD, ask for them.
5. Final answer: present the report sections - score, matched skills, missing skills, relevant experience,
   major gaps, recommended resume improvements, suggested learning areas, overall summary.

RULES: never invent experience the candidate does not have; only suggest edits they can honestly make;
the score is a keyword-overlap indicator, not a hiring prediction; do not promise job outcomes."""


DEFAULT_MODEL = "openrouter/free"   # OpenRouter's free router; set OPENROUTER_MODEL to pin a specific model

MISSING_KEY_MSG = ("OPENROUTER_API_KEY is not set.\n"
                   "  1) copy .env.example to .env  2) paste your key from https://openrouter.ai/keys\n"
                   "  (or run with --offline to test without a key).")


class MissingKeyError(RuntimeError):
    pass


def model_name() -> str:
    return os.getenv("OPENROUTER_MODEL", "").strip() or DEFAULT_MODEL


def build_llm(offline: bool):
    """Real LLM through OpenRouter (default). offline=True returns the rule-based test stand-in."""
    if offline:
        from offline_llm import ScriptedChatModel
        return ScriptedChatModel()
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not key or key.startswith("your-"):
        raise MissingKeyError(MISSING_KEY_MSG)
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=model_name(), api_key=key, base_url="https://openrouter.ai/api/v1", temperature=0)


def build_agent(llm):
    memory = MemorySaver()  # conversation memory: keeps all messages + tool results per thread_id
    try:                    # LangChain >= 1.0
        from langchain.agents import create_agent
        return create_agent(llm, ALL_TOOLS, system_prompt=SYSTEM_PROMPT, checkpointer=memory)
    except ImportError:     # older LangGraph
        from langgraph.prebuilt import create_react_agent
        return create_react_agent(llm, ALL_TOOLS, prompt=SYSTEM_PROMPT, checkpointer=memory)


def run_turn(agent, text, config, tracer, first=False, debug=False):
    """One user turn. Never raises: problems become a readable message."""
    tracer.start_turn(text, first)
    failed = False
    try:
        result = agent.invoke({"messages": [("user", text)]}, config)
        answer = result["messages"][-1].content
        answer = answer if isinstance(answer, str) else str(answer)
    except GraphRecursionError:
        failed, answer = True, "The agent used too many steps without finishing. Please try a simpler request."
    except Exception as e:  # noqa: BLE001
        if debug:
            traceback.print_exc()
        failed = True
        answer = (f"The agent run failed ({type(e).__name__}: {str(e)[:200]}).\n"
                  "Check your API key, that OPENROUTER_MODEL supports tool calling, and rate limits, then retry.")
    tracer.final(answer, failed)
    return answer


def validate_inputs(resume, jd):
    problems = []
    for label, p in (("resume", resume), ("job description", jd)):
        path = Path(p)
        if not path.is_file():
            problems.append(f"{label} file not found: {p}")
        elif path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            problems.append(f"{label} must be one of {', '.join(SUPPORTED_EXTENSIONS)} (got '{path.suffix}')")
    return problems


def make_goal(resume, jd):
    """The goal sent to the agent. The paths are whatever the user supplied (CLI args or UI uploads)."""
    return (f"Analyze my resume at '{resume}' against the job description at '{jd}'. "
            f"Tell me how well they match and how I can improve.")


class AnalysisSession:
    """One agent + one conversation memory (thread). Used by the web UI: one per browser session.

    analyze() sends the goal with the user's own files; ask() sends follow-ups on the SAME thread so the
    MemorySaver gives the agent the earlier resume/JD analysis.
    """

    def __init__(self, llm, logfile="logs/agent_trace.txt", header=""):
        self.agent = build_agent(llm)
        self.tracer = TraceHandler(logfile, header=header, quiet=True)
        self.config = {"configurable": {"thread_id": str(uuid.uuid4())},
                       "callbacks": [self.tracer], "recursion_limit": 25}

    def analyze(self, resume_path, jd_path):
        return run_turn(self.agent, make_goal(resume_path, jd_path), self.config, self.tracer, first=True)

    def ask(self, question):
        return run_turn(self.agent, question, self.config, self.tracer)

    def close(self):
        self.tracer.close()


def main():
    load_dotenv()
    ap = argparse.ArgumentParser(description="Job Application Assistant agent")
    ap.add_argument("--resume", default="data/sample_resume.txt", help="resume file (.txt/.md/.pdf/.docx); default = demo sample")
    ap.add_argument("--jd", default="data/sample_jd.txt", help="job description file; default = demo sample")
    ap.add_argument("--offline", action="store_true", help="use the rule-based stand-in model (no API key)")
    ap.add_argument("--ask", action="append", default=[], help="follow-up question(s); skips interactive chat")
    ap.add_argument("--log", default="logs/agent_trace.txt")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()

    problems = validate_inputs(a.resume, a.jd)
    if problems:
        sys.exit("ERROR: " + "\n       ".join(problems))

    try:
        llm = build_llm(a.offline)
    except MissingKeyError as e:
        sys.exit(f"ERROR: {e}")
    agent = build_agent(llm)
    tracer = TraceHandler(a.log, header="LLM: " + ("OFFLINE rule-based stand-in (not a real LLM)" if a.offline
                                                  else model_name()))
    config = {"configurable": {"thread_id": str(uuid.uuid4())}, "callbacks": [tracer], "recursion_limit": 25}

    goal = make_goal(a.resume, a.jd)
    run_turn(agent, goal, config, tracer, first=True, debug=a.debug)

    for q in a.ask:
        run_turn(agent, q, config, tracer, debug=a.debug)
    if not a.ask and sys.stdin.isatty():
        print("\nAsk follow-up questions (conversation memory is kept). Type 'exit' to quit.")
        while True:
            try:
                q = input("\nYou> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in ("", "exit", "quit"):
                break
            run_turn(agent, q, config, tracer, debug=a.debug)
    tracer.close()


if __name__ == "__main__":
    main()
