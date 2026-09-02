# src/hirelens/crew.py
# ==========================================
# THE ORCHESTRATOR — connects all 5 agents and their tasks
#
# INTERVIEW EXPLANATION:
#   "crew.py is the heart of a CrewAI project. It reads agent
#    and task definitions from YAML, connects them with tools,
#    and defines how they work together. The @agent decorator
#    tells CrewAI this method creates an agent. The @task
#    decorator tells CrewAI this method creates a task.
#    The @crew decorator assembles everything into one pipeline."
#
# CrewAI CORE CONCEPTS used here:
#   @agent  → method that returns an Agent object
#   @task   → method that returns a Task object
#   @crew   → method that returns the final Crew
#   Process.sequential → agents run one after another in order
# ==========================================

import os
import json
from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, task, crew
from langchain_ollama import OllamaLLM
from dotenv import load_dotenv

load_dotenv()

# ── Connect to Ollama ──
# This is the LLM ALL 5 agents will use
# We define it once here and pass it to each agent
from crewai import LLM
...
ollama_llm = LLM(
    model="ollama/mistral",
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
)


@CrewBase
class HirelensCrewAI:
    """
    The main CrewAI class for HireLens.

    @CrewBase decorator tells CrewAI:
      - Look for agents.yaml in config/agents.yaml
      - Look for tasks.yaml  in config/tasks.yaml
      - Wire everything together automatically

    INTERVIEW EXPLANATION:
        "CrewBase is a base class that handles all the YAML loading
         and agent/task registration automatically. I just define
         methods decorated with @agent and @task and CrewAI
         handles the rest."
    """

    # These tell CrewAI where to find the YAML files
    agents_config = "config/agents.yaml"
    tasks_config  = "config/tasks.yaml"

    # ─────────────────────────────────────────────
    # AGENTS — one method per agent
    # Each method reads from agents.yaml and returns an Agent object
    # ─────────────────────────────────────────────

    @agent
    def resume_analyst(self) -> Agent:
        """
        Agent 1 — Resume Intelligence Agent
        Reads: agents.yaml → resume_analyst section
        """
        return Agent(
            config=self.agents_config["resume_analyst"],
            llm=ollama_llm,
            verbose=True,       # prints what agent is thinking
            allow_delegation=False  # this agent works alone, no delegation
        )

    @agent
    def interview_strategist(self) -> Agent:
        """
        Agent 2 — Interview Planning Agent
        Reads: agents.yaml → interview_strategist section
        """
        return Agent(
            config=self.agents_config["interview_strategist"],
            llm=ollama_llm,
            verbose=True,
            allow_delegation=False
        )

    @agent
    def ai_interviewer(self) -> Agent:
        """
        Agent 3 — Chat Interview Agent
        Reads: agents.yaml → ai_interviewer section
        """
        return Agent(
            config=self.agents_config["ai_interviewer"],
            llm=ollama_llm,
            verbose=True,
            allow_delegation=False
        )

    @agent
    def answer_evaluator(self) -> Agent:
        """
        Agent 4 — Decision and Control Agent
        Reads: agents.yaml → answer_evaluator section
        """
        return Agent(
            config=self.agents_config["answer_evaluator"],
            llm=ollama_llm,
            verbose=True,
            allow_delegation=False
        )

    @agent
    def report_generator(self) -> Agent:
        """
        Agent 5 — Evaluation and Scoring Agent
        Reads: agents.yaml → report_generator section
        """
        return Agent(
            config=self.agents_config["report_generator"],
            llm=ollama_llm,
            verbose=True,
            allow_delegation=False
        )

    # ─────────────────────────────────────────────
    # TASKS — one method per task
    # Each method reads from tasks.yaml and returns a Task object
    # ─────────────────────────────────────────────

    @task
    def parse_resume_task(self) -> Task:
        """
        Task 1 — Parse the resume PDF text
        Agent: resume_analyst
        """
        return Task(
            config=self.tasks_config["parse_resume_task"]
        )

    @task
    def plan_interview_task(self) -> Task:
        """
        Task 2 — Generate tailored questions + rubric
        Agent: interview_strategist
        Depends on: parse_resume_task output
        """
        return Task(
            config=self.tasks_config["plan_interview_task"]
        )

    @task
    def conduct_interview_task(self) -> Task:
        """
        Task 3 — Generate interviewer message
        Agent: ai_interviewer
        This task runs multiple times (once per question)
        """
        return Task(
            config=self.tasks_config["conduct_interview_task"]
        )

    @task
    def evaluate_answer_task(self) -> Task:
        """
        Task 4 — Evaluate candidate answer
        Agent: answer_evaluator
        This task runs multiple times (once per answer)
        """
        return Task(
            config=self.tasks_config["evaluate_answer_task"]
        )

    @task
    def generate_report_task(self) -> Task:
        """
        Task 5 — Generate final hiring report
        Agent: report_generator
        Runs once at the end of the interview
        """
        return Task(
            config=self.tasks_config["generate_report_task"]
        )

    # ─────────────────────────────────────────────
    # CREW FACTORIES
    # Different crews for different stages of the interview
    #
    # WHY MULTIPLE CREWS?
    #   The interview has different phases:
    #   1. Setup phase    → parse resume + plan questions (runs once)
    #   2. Interview loop → conduct + evaluate (runs per answer)
    #   3. Report phase   → generate report (runs once at end)
    #
    # Each phase needs different agents/tasks
    # so we create separate crews for each phase.
    # ─────────────────────────────────────────────

    @crew
    def setup_crew(self) -> Crew:
        """
        SETUP CREW — runs once when resume is uploaded
        Agents: resume_analyst → interview_strategist
        Tasks:  parse_resume   → plan_interview

        INTERVIEW EXPLANATION:
            "Process.sequential means tasks run in order —
             Agent 1 finishes, then Agent 2 starts using
             Agent 1's output. This is a pipeline pattern."
        """
        return Crew(
            agents=[
                self.resume_analyst(),
                self.interview_strategist()
            ],
            tasks=[
                self.parse_resume_task(),
                self.plan_interview_task()
            ],
            process=Process.sequential,
            verbose=True
        )

    @crew
    def interview_crew(self) -> Crew:
        """
        INTERVIEW CREW — runs once per candidate answer
        Agents: ai_interviewer → answer_evaluator
        Tasks:  conduct        → evaluate
        """
        return Crew(
            agents=[
                self.ai_interviewer(),
                self.answer_evaluator()
            ],
            tasks=[
                self.conduct_interview_task(),
                self.evaluate_answer_task()
            ],
            process=Process.sequential,
            verbose=True
        )

    @crew
    def report_crew(self) -> Crew:
        """
        REPORT CREW — runs once when interview ends
        Agents: report_generator
        Tasks:  generate_report
        """
        return Crew(
            agents=[self.report_generator()],
            tasks=[self.generate_report_task()],
            process=Process.sequential,
            verbose=True
        )