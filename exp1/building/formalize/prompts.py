from __future__ import annotations

# ── Stage 1: Task Decomposition ──
STAGE1_SYSTEM = """You are a task decomposition expert. Your job is to break down a complex task into clear sub-tasks for a multi-agent system.

Output XML with this EXACT structure:

<task_decomposition>
  <goal>one-sentence summary of the final answer to compute</goal>
  <sub_tasks>
    <sub_task idx="0">description of sub-task 0</sub_task>
    <sub_task idx="1">description of sub-task 1</sub_task>
  </sub_tasks>
</task_decomposition>

CRITICAL RULES:
- Number sub-tasks sequentially starting from idx="0"
- Each sub-task should be a self-contained unit of work
- Output 3-6 sub-tasks
- Output ONLY the XML, no markdown fences, no extra text"""

STAGE1_TEMPLATE = """Task: {task_text}

{dataset_requirement}

Break this task into sub-tasks. Each sub-task should be a clear, self-contained step that one agent can solve. Output XML only."""

# ── Stage 2: Agent Role Assignment ──
STAGE2_SYSTEM = """You are a multi-agent role engineer. Given a task decomposition, assign appropriate agent roles.

Output XML with this EXACT structure:

<agent_specs>
  <agents>
    <agent id="a0" is_final="false" execution_mode="llm_only">
      <role>detailed role description for this agent</role>
      <sub_task_idx>0</sub_task_idx>
      <dependencies>a1,a2</dependencies>
    </agent>
    <agent id="a1" is_final="true" execution_mode="llm_only">
      <role>role for the final agent that produces the answer</role>
      <sub_task_idx>1</sub_task_idx>
      <dependencies></dependencies>
    </agent>
  </agents>
</agent_specs>

CRITICAL RULES:
- Exactly one agent must have is_final="true"
- The final agent should aggregate results and produce the final answer
- Each agent handles one sub-task (match by sub_task_idx)
- Set execution_mode explicitly on every agent. Use only llm_only or multi_turn_search.
- Output ONLY the XML, no markdown fences, no extra text"""

STAGE2_TEMPLATE = """Task: {task_text}

Task Decomposition:
<task_decomposition>
  <goal>{goal}</goal>
  <sub_tasks>
{sub_tasks_xml}
  </sub_tasks>
</task_decomposition>

{dataset_requirement}

Assign agent roles for each sub-task. The final agent must produce the answer in the required format. Output XML only."""


def get_dataset_requirement(dataset: str) -> str:
    if dataset == "aime":
        return "IMPORTANT: The final answer MUST be a single integer. Output it as \\boxed{answer}."
    elif dataset == "drb":
        return "IMPORTANT: The final answer should be a comprehensive research article with proper citations and structure."
    elif dataset == "bcp":
        return (
            "IMPORTANT: The final answer should be a single concise answer string (e.g., a name, date, or number). "
            "Do NOT include extra explanation. Route each agent by whether it needs NEW corpus evidence: use "
            "execution_mode=\"multi_turn_search\" only for evidence discovery/retrieval, and "
            "execution_mode=\"llm_only\" for reasoning over upstream evidence, reconciliation, verification "
            "without new facts, final synthesis, and exact-output formatting. Do not make every agent search. "
            "Include at least one multi_turn_search agent, and normally make the final agent llm_only."
        )
    return ""
