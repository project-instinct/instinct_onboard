# Project AGENTS Guidelines

> **Read first:** [agents/CommonAgentInstructions.md](agents/CommonAgentInstructions.md) contains the shared baseline — core rules, working style (before/during/after editing), Linus's coding taste, and the ask-before-proceeding checklist. This file adds project-specific overrides and architecture notes. When the two conflict, this file takes precedence.

## Environment

- You are running on an **Agent-Host computer** — do not run commands directly; ask for how to connect and run commands on the run-host if not known.
- Do not change files outside the project folder on the run-host without explicit permission.
- Do not change any comments unless they become inconsistent with the current design.
- Always run `pre-commit run --all-files` before committing to ensure code quality and consistency.

## Real World Experiment

Since this is a deployment code for instinctlab workflow, the "real world" is the physical robot. The "simulated world" is the simulation environment. The codebase should be designed to run on the real robot with minimal changes, and all simulation-specific code should be well-abstracted.

- For safety, the program (especially the ros_nodes) side, must have a "dry-run" mode by default to test the code without sending actual commands to the robot.
- The entry script must have implementation on user-side emergency stop (e-stop) mechanism, which can be as simple as listening to a specific joystick button and shutting down the motors immediately when pressed.
- If not production-critical, quick features must be implemented in the entry script instead of the agent or ros_node, to keep the core logic clean and focused. The entry script is the best place for wiring up different components together, such as connecting the joystick input to the velocity command buffer.
