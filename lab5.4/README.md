# day-5-capstone-lab

**Estimated Time:** 150 minutes

**Lab Objective:**
Develop an AI-driven solution strategy — architecture, working model, deployment plan, and business case — tailored to a specific business problem.

**Scenario:**

Your healthcare client is facing challenges in accurately predicting patient readmissions. Your task is to develop an AI-based solution to address this problem, and prepare a proposal that clearly articulates the expected business value, backed by a working model and a documented rollout plan.

**Learning Objectives:**

In this capstone lab, you will:
- Define the business problem, stakeholders, and success metrics.
- Select an AI architecture and justify the choice in an **Architecture Decision Record (ADR)**.
- Use Hospital Readmission data to build and evaluate a working model that predicts readmission risk.
- Translate model performance into a **business justification** (cost of readmission, expected savings, ROI framing).
- Draft a phased **implementation roadmap**, including a deployment plan and risk mitigation strategies.
- Create a 5-minute pitch to deliver for a client or internal stakeholders (optional)

**Time Breakdown:**

| Part | Focus | Est. Time |
| --- | --- | --- |
| Part 1 | Business Problem Framing | 15 min |
| Part 2 | Data Loading, EDA & Preprocessing | 35 min |
| Part 3 | Model Building & Architecture Decision Record | 40 min |
| Part 4 | Model Evaluation & Business Justification | 25 min |
| Part 5 | Implementation Roadmap | 20 min |
| Wrap-Up | Stakeholder Pitch Prep (optional) | 15 min |
| **Total** | | **~150 min** |

> The dataset used in this lab comes
> from [kaggle.com](https://www.kaggle.com/code/iabhishekofficial/prediction-on-hospital-readmission/input)

**Here is the [SA-docker](https://github.com/ga-curriculum-custom/deloitte-ai-solution-architect-v2.0-q1-2026/tree/main/modules/sa-docker) container repo for reference.**


## How Your Capstone Will Be Reviewed

Your team's work is assessed against the five parts of the lab. Use this as a checklist while you work — if every row reads "Meets" before you finish, you're in good shape.

| Criterion | Exceeds | Meets | Developing |
| --- | --- | --- | --- |
| **1. Problem framing & success metrics** (Part 1) | Problem, stakeholders, and success metrics are specific and measurable; metrics tie directly to readmission outcomes | Problem and stakeholders defined; at least one measurable success metric | Problem stated generally; metrics vague or missing |
| **2. Architecture decision** (Part 3) | Decision record weighs two or more realistic alternatives with clear trade-offs; choice follows from the evidence | Decision record documents the chosen approach and at least one alternative | Approach chosen without documented reasoning or alternatives |
| **3. Working model & evaluation** (Parts 2–4) | Model runs end to end; evaluation uses appropriate metrics and the team can explain what they mean | Model runs; evaluation present with standard metrics | Model incomplete, or evaluation missing or misread |
| **4. Business justification** (Part 4) | Model performance translated into cost/savings terms a non-technical stakeholder could act on | Business value stated with at least a rough cost or ROI framing | Technical results reported without business translation |
| **5. Implementation roadmap** (Part 5) | Phased plan with deployment approach, key risks, and mitigations | Phased plan with deployment approach | Next steps listed without phasing or risk consideration |
| **Bonus: 5-minute pitch** (optional) | Clear, timed, persuasive — a client would say yes to a next meeting | Covers the five beats in roughly five minutes | Optional — no penalty if skipped |

## Your 5-Minute Pitch (Optional Wrap-Up)

Good news: you've already built every piece of this pitch — the wrap-up is assembly, not creation. One minute per beat:

1. **The problem** *(from Part 1)* — who's hurt by readmissions, and what does success look like?
2. **Your recommended approach** *(from Part 3)* — what you chose and, briefly, why not the alternative. Your Architecture Decision Record is your script.
3. **The evidence** *(from Part 4)* — what the model showed. One or two numbers, not a metrics tour.
4. **The business value** *(from Part 4)* — what those numbers mean in dollars or outcomes for the client.
5. **The ask** *(from Part 5)* — what happens next: phase one of your roadmap, and what you need to start it.

> Tip: assign one beat per team member and you've also solved who-presents-what.


<!--
### Lab Setup Instructions
1. **Navigate to the Project Directory:**
   ```bash
   cd day-5-capstone-lab
   ```

2. **Create a Virtual Environment:**
   ```bash
   python3 -m venv env
   ```

3. **Activate the Virtual Environment:**
   ```bash
   source env/bin/activate
   ```

4. **Install the Required Packages:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

6. **Open the Starter IPython Notebook:**
    - In the Jupyter interface, navigate to the starter notebook to begin the lab.
-->