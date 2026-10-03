"""Observable execution trace.

Logs WHAT the agent does (tool selected, tool input, observation, final answer) - not the model's
hidden reasoning. Output goes to the console and to a log file.
"""
import json
import os
from datetime import datetime
from pathlib import Path
from langchain_core.callbacks import BaseCallbackHandler


# milestone label shown in the UI for each tool observation
MILESTONES = {
    "load_documents": "Documents loaded",
    "analyze_skill_gap": "Skill gaps analyzed",
    "calculate_match_score": "Match score calculated",
    "generate_report": "Report generated",
}


class TraceHandler(BaseCallbackHandler):
    """Writes the trace to a log file (and the console unless quiet=True) and also keeps it as a list of
    structured `events` so the web UI can display it. Only actions/observations are recorded."""

    def __init__(self, logfile="logs/agent_trace.txt", header="", quiet=False):
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        self.f = open(logfile, "a", encoding="utf-8")
        self.quiet = quiet
        self.events = []                                 # [{"turn","step","kind","tool","label","detail","ok"}]
        self.turn = 0
        self.step, self.calls, self.notes = 0, [], []   # self.calls = tools used (handy for tests)
        self._w("=" * 72)
        self._w(f"AGENT SESSION  {datetime.now():%Y-%m-%d %H:%M:%S}   {header}")
        self._w("=" * 72)

    def _w(self, text=""):
        if not self.quiet:
            print(text, flush=True)
        self.f.write(text + "\n"); self.f.flush()

    def _ev(self, kind, label, detail="", tool="", ok=True):
        self.events.append({"turn": self.turn, "step": self.step, "kind": kind, "tool": tool,
                            "label": label, "detail": detail, "ok": ok})

    # ---- called by agent.py
    def start_turn(self, text, first):
        self.step, self.calls, self.notes = 0, [], []
        self.turn += 1
        self._w(f"\n{'Goal received' if first else 'Follow-up received'}:\n  {text}")
        self._ev("goal", "Goal received" if first else "Follow-up received", text[:200])

    def final(self, text, failed=False):
        self._w("\nFinal response to user:\n" + "\n".join("  " + l for l in text.splitlines()))
        none = "none - the run failed" if failed else "none - answered from memory"
        self._w(f"\n(tools used this turn: {', '.join(self.calls) or none})")
        if not self.calls and not failed:
            self._ev("memory", "Answered from memory", "No tool call was needed")
        self._ev("final", "Run failed" if failed else "Final response prepared", ok=not failed)

    # ---- LangChain callbacks
    def on_llm_end(self, response, **kw):
        msg = response.generations[0][0].message
        note = msg.content if isinstance(msg.content, str) else ""
        self.notes.append(note.strip()[:160] if getattr(msg, "tool_calls", None) else "")

    def on_tool_start(self, serialized, input_str, **kw):
        self.step += 1
        name = kw.get("name") or (serialized or {}).get("name", "?")
        args = kw.get("inputs") if isinstance(kw.get("inputs"), dict) else input_str
        self.calls.append(name)
        self._w(f"\nStep {self.step}:")
        if self.notes and self.notes[-1]:
            self._w(f"  Agent decision : {self.notes[-1]}")
        self._w(f"  Tool selected  : {name}")
        shown = (json.dumps(args) if isinstance(args, dict) else str(args))[:220]
        self._w(f"  Tool input     : {shown}")
        self._ev("tool_call", f"Tool selected: {name}", shown, tool=name)

    def on_tool_end(self, output, **kw):
        raw = getattr(output, "content", output)
        try:
            summary = json.loads(raw)["summary"]
        except Exception:  # noqa: BLE001
            summary = str(raw)[:300]
        self._w(f"  Observation    : {summary}")
        name = self.calls[-1] if self.calls else ""
        try:
            ok = json.loads(raw).get("status") == "ok"
        except Exception:  # noqa: BLE001
            ok = False
        self._ev("observation", MILESTONES.get(name, name) if ok else f"{name} failed", summary, tool=name, ok=ok)

    def on_tool_error(self, error, **kw):
        self._w(f"  Observation    : TOOL ERROR - {error}")
        self._ev("observation", "Tool error", str(error)[:300], tool=self.calls[-1] if self.calls else "", ok=False)

    def close(self):
        self.f.close()
