"""Offline stand-in for the LLM (used by `--offline` and by tests/run_tests.py).

IMPORTANT: this is NOT a real language model. It is a tiny rule-based policy that looks at the
conversation so far (user message + earlier tool observations) and returns the next tool call or a
final answer, in the same message format a real tool-calling LLM uses. It exists so the whole
pipeline - tools, memory, tracing, error handling - can be tested without an API key or internet.
Real runs use OpenRouter (see agent.py). Its wording is template-based, not generated.
"""
import json
import re
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

PATH_RE = re.compile(r"'([^']+\.(?:txt|pdf|md|docx))'", re.I)


def _j(msg):
    try:
        return json.loads(msg.content)
    except Exception:  # noqa: BLE001
        return {"status": "error", "summary": str(msg.content)}


class ScriptedChatModel(BaseChatModel):
    @property
    def _llm_type(self):
        return "offline-rule-based-stand-in"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        last_h = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
        user = messages[last_h].content
        turn = [m for m in messages[last_h + 1:] if isinstance(m, ToolMessage)]
        history = [m for m in messages if isinstance(m, ToolMessage)]
        cid = f"call_{len(messages)}"

        def call(name, args, note):
            return AIMessage(content=note, tool_calls=[{"name": name, "args": args, "id": cid}])

        def say(text):
            return AIMessage(content=text)

        paths = PATH_RE.findall(user)
        if not turn:
            if len(paths) >= 2:
                msg = call("load_documents", {"resume_path": paths[0], "jd_path": paths[1]},
                           "I need the resume and job description before any analysis.")
            else:
                msg = self._follow_up(user, history, say)
        else:
            last = _j(turn[-1])
            name = getattr(turn[-1], "name", "")
            if last.get("status") == "error":
                msg = say(f"I could not finish the analysis. {last['summary']}\n"
                          "Please check the file paths / tool problem above and try again.")
            elif name == "load_documents":
                msg = call("analyze_skill_gap", {}, "Documents are loaded; next I compare skills.")
            elif name == "analyze_skill_gap":
                msg = call("calculate_match_score", {}, "Gap analysis is done; next I compute the score.")
            elif name == "calculate_match_score":
                gap = next(_j(m) for m in turn if getattr(m, "name", "") == "analyze_skill_gap")
                miss, pref = gap["missing_skills"], gap["preferred_missing"]
                recs = [f"Add a project or coursework bullet that shows hands-on {s}." for s in miss[:4]] \
                    or ["Your required skills are covered; quantify results in your project bullets."]
                learn = [f"{s}: take an introductory tutorial, then use it in a small project." for s in (miss + pref)[:5]]
                summ = (f"Your resume covers {len(gap['matched_skills'])} of {len(gap['required_skills'])} required skills. "
                        f"The main gaps are {', '.join(miss) or 'none'}. Closing these with honest, project-backed "
                        f"evidence is the most useful next step.")
                msg = call("generate_report", {"recommendations": recs, "learning_areas": learn, "summary": summ},
                           "I have the score; now I write the improvement report.")
            elif name == "generate_report":
                msg = say("Analysis complete. The report was saved to outputs/improvement_report.md.\n\n"
                          + last["report_markdown"])
            else:
                msg = say(last.get("summary", "Done."))
        return ChatResult(generations=[ChatGeneration(message=msg)])

    @staticmethod
    def _follow_up(user, history, say):
        gap = next((_j(m) for m in reversed(history) if getattr(m, "name", "") == "analyze_skill_gap"
                    and _j(m).get("status") == "ok"), None)
        score = next((_j(m) for m in reversed(history) if getattr(m, "name", "") == "calculate_match_score"
                      and _j(m).get("status") == "ok"), None)
        if not gap:
            return say("I don't have an analysis in memory yet. Please give me a resume file and a job "
                       "description file (e.g. python agent.py --resume cv.txt --jd jd.txt).")
        q = user.lower()
        if "score" in q and score:
            return say(f"Your {score['summary']}")
        if "gap" in q or "missing" in q or "lack" in q:
            n = 3 if "three" in q or "3" in q else 5 if "five" in q or "5" in q else len(gap["missing_skills"])
            top = gap["missing_skills"][:n]
            return say("From the analysis I already did (no need to re-read the files), your biggest gaps are: "
                       + ", ".join(top) + ". They are listed in the order the job description mentions them.")
        return say("Using my earlier analysis: " + (score["summary"] if score else "")
                   + " Missing skills: " + (", ".join(gap["missing_skills"]) or "none") + ".")
