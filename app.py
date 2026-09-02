# app.py — Flask bridge between browser and CrewAI agents
import os, uuid, json
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import PyPDF2
from src.hirelens.crew import HirelensCrewAI

load_dotenv()
app = Flask(__name__)
SESSIONS = {}  # stores each candidate's interview data


def get_pdf_text(file):
    reader = PyPDF2.PdfReader(file)
    return "\n".join(p.extract_text() or "" for p in reader.pages)


def parse_json(raw):
    """Extract JSON from LLM's text output."""
    raw = raw.replace("```json", "").replace("```", "")
    try:
        return json.loads(raw[raw.find("{"):raw.rfind("}")+1])
    except:
        return {}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload():
    """Agent 1 + Agent 2: parse resume, plan questions."""
    file = request.files["file"]
    role = request.form.get("role", "Software Engineer")
    pdf_text = get_pdf_text(file)

    crew = HirelensCrewAI()
    result = crew.setup_crew().kickoff(inputs={
        "role": role, "resume_text": pdf_text[:6000], "resume_data": ""
    })

    resume = parse_json(result.tasks_output[0].raw)
    plan   = parse_json(result.tasks_output[1].raw)
    questions = plan.get("questions") or [f"Tell me about your experience for {role}."]
    rubric    = plan.get("rubric", "")
    name      = resume.get("name", "Candidate")

    # Agent 3: opening message
    opening = crew.interview_crew().kickoff(inputs={
        "role": role, "candidate_name": name, "conversation_history": "",
        "current_question": questions[0], "mode": "opening",
        "question": questions[0], "answer": "", "attempt": "1", "rubric": rubric
    }).tasks_output[0].raw

    sid = str(uuid.uuid4())
    SESSIONS[sid] = {
        "role": role, "resume": resume, "name": name,
        "questions": questions, "rubric": rubric,
        "history": [{"role": "interviewer", "text": opening}],
        "q_index": 0, "attempt": 1, "scores": []
    }

    return jsonify({"session_id": sid, "resume": resume,
                     "questions": questions, "opening_message": opening})


@app.route("/api/respond", methods=["POST"])
def respond():
    """Agent 3 + Agent 4: respond to candidate, evaluate answer."""
    d = request.json
    s = SESSIONS[d["session_id"]]
    s["history"].append({"role": "candidate", "text": d["message"]})

    q = s["questions"][s["q_index"]]
    history_text = "\n".join(f"{t['role']}: {t['text']}" for t in s["history"][-6:])

    result = HirelensCrewAI().interview_crew().kickoff(inputs={
        "role": s["role"], "candidate_name": s["name"],
        "conversation_history": history_text, "current_question": q,
        "mode": "next", "question": q, "answer": d["message"],
        "attempt": str(s["attempt"]), "rubric": s["rubric"]
    })

    message = result.tasks_output[0].raw
    eval_data = parse_json(result.tasks_output[1].raw)
    decision = eval_data.get("decision", "NEXT").upper()
    score = max(1, min(10, int(eval_data.get("score", 5))))

    done = False
    if decision == "FOLLOW_UP" and s["attempt"] < 2:
        s["attempt"] = 2
    else:
        s["scores"].append(score)
        s["q_index"] += 1
        s["attempt"] = 1
        done = s["q_index"] >= len(s["questions"])

    s["history"].append({"role": "interviewer", "text": message})

    return jsonify({"message": message, "decision": decision,
                     "score": score, "interview_done": done})


@app.route("/api/finish", methods=["POST"])
def finish():
    """Agent 5: generate final report."""
    s = SESSIONS[request.json["session_id"]]
    transcript = "\n".join(f"{t['role']}: {t['text']}" for t in s["history"])
    avg = round(sum(s["scores"]) / len(s["scores"]), 1) if s["scores"] else 0

    result = HirelensCrewAI().report_crew().kickoff(inputs={
        "candidate_name": s["name"], "role": s["role"],
        "transcript": transcript[:4000], "scores": str(s["scores"]),
        "average_score": str(avg), "rubric": s["rubric"][:1500],
        "skills": s["resume"].get("skills", ""),
        "experience": s["resume"].get("experience", "")
    })

    report = parse_json(result.tasks_output[0].raw)
    report.setdefault("final_verdict", "Hire" if avg >= 6 else "No Hire")
    report.setdefault("overall_score", int(avg * 10))
    return jsonify(report)


if __name__ == "__main__":
    print("HireLens CrewAI running → http://127.0.0.1:5000")
    app.run(debug=True, port=5000)