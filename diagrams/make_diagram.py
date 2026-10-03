import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(12, 7.8)); ax.set_xlim(0, 12); ax.set_ylim(0, 7.8); ax.axis("off")
NAVY, BLUE, ORG, GRN = "#1f3b73", "#2f6fd6", "#d9730d", "#2a8a2a"

def box(x, y, w, h, text, fc, fs=10, bold=False, ec="#333", color="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12", fc=fc, ec=ec, lw=1.4))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", color=color)

def arrow(p, q, color="#333", ls="-", lw=1.8):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=15, color=color, lw=lw, linestyle=ls))

ax.text(6, 7.55, "Job Application Assistant - Agent Architecture", ha="center", fontsize=14, fontweight="bold", color=NAVY)

# agent loop strip
steps = ["1 Goal", "2 Planning", "3 Tool Call", "4 Observation", "5 Next Action", "6 Final Output"]
for i, s in enumerate(steps):
    box(0.35 + i*1.9, 6.65, 1.65, 0.5, s, "#eef3ff" if i not in (0, 5) else "#d9f2d9", fs=9, bold=True)
    if i < 5: arrow((2.0 + i*1.9, 6.9), (2.25 + i*1.9, 6.9), NAVY, lw=1.3)
ax.text(6.0, 6.35, "steps 3 - 5 repeat in a loop until the goal is complete", fontsize=8.8, color=ORG, ha="center", style="italic")

# user
box(0.2, 4.15, 1.6, 1.3, "WEB UI\n(app.py)\n\nresume upload\n+ JD upload/paste", "#d9f2d9", fs=8.3, bold=True)
# agent
box(2.5, 3.85, 5.9, 1.9, "", "#dbe9ff", ec=NAVY)
ax.text(5.45, 5.4, "LLM AGENT  (Orchestrator)", ha="center", fontsize=11.5, fontweight="bold", color=NAVY)
ax.text(5.45, 4.65, "Understands the goal  -  plans the next step  -  chooses a tool\nreads the observation  -  decides the next action  -  writes the answer\n\nLLM via OpenRouter  |  LangChain / LangGraph (ReAct-style tool calling)",
        ha="center", va="center", fontsize=8.8)
# memory
box(9.0, 3.85, 2.8, 1.9, "AGENT MEMORY\n\nConversation memory\n(MemorySaver: messages\n+ tool results)\n\nWorking memory (SESSION)\ndocs read once, shared\nby all tools", "#fff3c4", fs=8.3, ec="#8a7000")
arrow((8.4, 4.8), (9.0, 4.8), "#8a7000"); arrow((9.0, 4.55), (8.4, 4.55), "#8a7000")

arrow((1.8, 4.8), (2.5, 4.8), GRN); ax.text(2.15, 5.12, "goal", fontsize=8, color=GRN, ha="center")
arrow((2.5, 4.35), (1.8, 4.35), GRN); ax.text(2.15, 4.1, "answer", fontsize=8, color=GRN, ha="center")

# tools
tools = [("1  Document Reader\nload_documents\n(.pdf / .docx / .txt)", 0.2), ("2  Skill Gap Analyzer\nanalyze_skill_gap\n(required/matched/missing)", 2.5),
         ("3  Match Score Calculator\ncalculate_match_score\n(matched / required x 100)", 4.8), ("4  Report Generator\ngenerate_report\n(structured .md report)", 7.1)]
for t, x in tools:
    box(x, 1.95, 2.1, 1.1, t, "#fff3c4", fs=8.2, ec="#8a7000")
    cx = x + 1.05; ax_x = min(max(cx, 2.8), 8.1)
    arrow((ax_x - 0.18, 3.85), (cx - 0.18, 3.05), BLUE)
    arrow((cx + 0.18, 3.05), (ax_x + 0.18, 3.85), ORG)
ax.text(0.25, 3.45, "tool call", color=BLUE, fontsize=8.5, fontweight="bold")
ax.text(0.25, 3.2, "observation", color=ORG, fontsize=8.5, fontweight="bold")

# final output and trace
box(2.3, 0.25, 5.2, 0.85, "FINAL OUTPUT\nscore - matched / missing skills - gaps - improvements\nshown in the web UI + downloadable report", "#d9f2d9", fs=8.3, bold=True)
arrow((8.0, 1.95), (6.9, 1.1), GRN); ax.text(8.1, 1.45, "report", fontsize=8, color=GRN)
box(9.5, 1.95, 2.3, 1.1, "TRACE LOGGER\nlogs every step:\ntool, input,\nobservation", "#eee", fs=8.3)
plt.savefig("diagrams/architecture_diagram.png", dpi=170, bbox_inches="tight", facecolor="white")
