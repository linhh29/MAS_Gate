---
name: bad_ensemble_skill
description: "A deliberately naive meta-agent skill for generating simplistic MAS designs. It avoids meaningful task decomposition, uses generic solver agents with no specialization, and assembles them in a flat ensemble with majority voting."
tags:
  - meta-agent
  - naive
  - ensemble
  - baseline
inputs:
  - user_query
---

## 1. Task Decomposition Module (The "What")
*Core Objective: Simply split the problem into independent parallel attempts without any meaningful decomposition.*

- **Intent & Scope Analysis:** The task is just "solve the problem". Do not analyze sub-steps or intermediate reasoning stages.
- **Sub-task Breakdown:** Create exactly 3 identical sub-tasks named "attempt_1", "attempt_2", "attempt_3". Each sub-task is simply "solve the given problem independently". No further decomposition.
- **Logical Dependency Mapping:** All three attempts are parallel and independent. No dependencies between them. The only dependency is: all three attempts must complete before the final aggregation step.
- **Success Criteria:** Each attempt should produce a final answer in \boxed{} format. The final aggregation should pick the majority answer.

## 2. Agent Engineering Module (The "Who")
*Core Objective: Design generic, non-specialized agents that all do the same thing.*

- **Role Profiling:** All three solver agents have the same generic role: "Problem Solver". No domain-specific specialization, no distinct perspectives, no complementary skills. They are all identical.
- **Instruction Design:** Each solver agent gets the same instruction: "You are a math problem solver. Read the problem carefully and solve it step by step. Put your final answer in \boxed{answer} format." No tailored instructions, no role-specific strategies.
- **Input Context Framing:** Each solver agent receives the exact same user query with no additional context or framing. The final aggregation agent receives all three outputs and simply picks the answer that appears most frequently (majority vote).

## 3. Workflow & Orchestration Module (The "How")
*Core Objective: Wire the agents into the simplest possible parallel-execute-then-vote topology.*

- **Architectural Topology:** Use a flat 3-way parallel topology. All three solver agents execute independently and concurrently. A final aggregation agent collects all outputs and performs majority voting. No iterative refinement, no feedback loops, no cross-agent communication.
- **Dataflow & State Management:** Each solver agent reads from the same global task input. The final aggregation agent receives all three solver outputs as raw text. No intermediate state transformation, no selective information routing.
- **Executable Generation:** Generate the code using the standard MASSequentialWorkflowTemplate (or MASMultiDependencyWorkflowTemplate) pattern. The execution order is: solver_1 → solver_2 → solver_3 → aggregator (sequential for simplicity). The final_output must be the aggregator's answer in \boxed{} format.