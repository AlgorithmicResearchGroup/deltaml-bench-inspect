# Vivaria Agent Scaffolding Guide

## Overview

This guide explains how to create and structure agents for the Vivaria AI benchmarking platform, based on analysis of the `modular-public` reference agent and practical experience integrating custom agents like the ARG agent.

## Table of Contents

1. [Vivaria Agent Architecture](#vivaria-agent-architecture)
2. [Core Requirements](#core-requirements)
3. [PyHooks Interface](#pyhooks-interface)
4. [Agent Structure Patterns](#agent-structure-patterns)
5. [Configuration and Manifest](#configuration-and-manifest)
6. [Implementation Best Practices](#implementation-best-practices)
7. [Common Integration Patterns](#common-integration-patterns)
8. [Troubleshooting](#troubleshooting)

---

## Vivaria Agent Architecture

### Basic Structure

Vivaria agents are arbitrary Python programs that communicate with the Vivaria platform through the `pyhooks` library. An agent consists of:

```
agent-directory/
├── main.py              # Entry point with hooks.main(async_main)
├── requirements.txt     # Python dependencies (must include pyhooks)
├── manifest.json        # Agent configuration and settings packs
└── ... (agent code)     # Your custom agent implementation
```

### Key Concepts

- **Agent**: The main controller that coordinates different modules
- **pyhooks**: Python library for Vivaria communication (LLM API, actions, logging)
- **Settings Packs**: Predefined configurations for different agent behaviors
- **Actions**: Standard operations (bash, python, score, submit, etc.)
- **State Management**: Persistent state across agent execution

---

## Core Requirements

### 1. Entry Point (`main.py`)

Every agent must have a `main.py` with this pattern:

```python
from pyhooks import Hooks

hooks = Hooks()

async def main(*args):
    # Your agent logic here
    task = await hooks.getTask()
    print(f"Task: {task.instructions}")
    
    # ... implement your solution ...
    
    # Submit result
    await hooks.submit("solution content")

if __name__ == "__main__":
    hooks.main(main)
```

### 2. Dependencies (`requirements.txt`)

Must include at minimum:

```
pyhooks
```

Plus any additional dependencies your agent needs.

### 3. Configuration (`manifest.json`)

Defines agent metadata and settings:

```json
{
    "defaultSettingsPack": "default",
    "settingsPacks": {
        "default": {
            "description": "Default agent configuration"
        }
    }
}
```

---

## PyHooks Interface

### Core Methods

```python
from pyhooks import Hooks

hooks = Hooks()

# Get task information
task = await hooks.getTask()
task.instructions  # Task description
task.scoring       # Scoring configuration

# Execute actions
result = await hooks.action("bash", command="ls -la")
result = await hooks.action("python", code="print('hello')")

# Submit solutions
await hooks.submit("final answer")

# Scoring (for intermediate evaluation)
score_result = await hooks.score()

# Logging and state
hooks.log("message")  # Log to Vivaria
hooks.save_state({"key": "value"})  # Persist state

# Usage tracking
usage = await hooks.get_usage()
usage.usage.tokens  # Current token usage
usage.usageLimits.tokens  # Token limits
```

### Environment Variables

Vivaria provides these environment variables:

- `TASK_ID`: Current task identifier
- `RUN_ID`: Current run identifier  
- `API_URL`: Vivaria server URL
- `AGENT_BRANCH_NUMBER`: Agent branch number

---

## Agent Structure Patterns

### 1. Simple Linear Agent

```python
async def main(*args):
    task = await hooks.getTask()
    
    # Analyze task
    analysis = await analyze_task(task.instructions)
    
    # Generate solution
    solution = await generate_solution(analysis)
    
    # Test solution
    test_result = await test_solution(solution)
    
    # Submit
    await hooks.submit(solution)
```

### 2. Modular Agent (like modular-public)

The `modular-public` agent uses a sophisticated 5-module architecture:

```python
# State-driven modular approach
class Agent:
    def __init__(self, state, settings, toolkit_dict):
        self.state = state
        self.settings = settings
        self.toolkit_dict = toolkit_dict

async def main(*args):
    task = await hooks.getTask()
    
    state = State(
        task_string=task.instructions,
        next_step={"module_type": "prompter", "args": {}},
        # ... other state
    )
    
    agent = Agent(state=state, settings=settings, toolkit_dict={})
    
    while True:
        if agent.state.next_step["module_type"] == "prompter":
            await prompter_module(agent)
        elif agent.state.next_step["module_type"] == "generator":
            await generator_module(agent)
        elif agent.state.next_step["module_type"] == "discriminator":
            await discriminator_module(agent)
        elif agent.state.next_step["module_type"] == "actor":
            await actor_module(agent)
        
        # Save state and check for completion
        hooks.save_state(agent.state.model_dump())
        await agent.autosubmit()
```

**Five Modules:**

1. **Prompter**: Prepares messages for LLM generation
2. **Generator**: Gets LLM responses via Vivaria's middleman
3. **Discriminator**: Selects best option from generations  
4. **Actor**: Executes function calls/actions
5. **Toolkit**: Defines available tools

### 3. Wrapper Agent (like ARG integration)

For integrating existing agents:

```python
from pyhooks import Hooks
from your_agent import YourExistingAgent

hooks = Hooks()

class VivariaWrapper:
    def __init__(self, task_instructions: str):
        self.task_instructions = task_instructions
        self.agent = YourExistingAgent(
            # Map Vivaria context to your agent's requirements
        )
        
    async def _vivaria_action(self, action_type: str, **kwargs):
        """Bridge your agent's tools to Vivaria actions"""
        return await hooks.action(action_type, **kwargs)
        
    async def solve(self) -> str:
        # Adapt your agent's solving method
        return await self.agent.solve_with_vivaria_bridge()

async def main(*args):
    task = await hooks.getTask()
    wrapper = VivariaWrapper(task.instructions)
    result = await wrapper.solve()
    await hooks.submit(result)

if __name__ == "__main__":
    hooks.main(main)
```

---

## Configuration and Manifest

### Manifest Structure

```json
{
    "defaultSettingsPack": "pack_name",
    "settingsPacks": {
        "pack_name": {
            "description": "Description of this configuration",
            "model": "gpt-4o-2024-05-13",
            "temperature": 0.1,
            "max_tokens": 4096,
            // For modular agents:
            "prompter": "_basic",
            "generator": "_gpt_4o", 
            "discriminator": "_basic",
            "actor": "_basic",
            "toolkit": "_basic"
        },
        "reasoning_pack": {
            "description": "Configuration optimized for reasoning tasks",
            "model": "o1-preview",
            "temperature": 0.0
        }
    }
}
```

### Settings Pack Naming Convention

The modular-public agent uses descriptive naming:
- `t_` prefix for toolkit
- `p_` for prompter  
- `g_` for generator
- `d_` for discriminator
- `a_` for actor

Example: `t_basic_p_context_aware_g_gpt_4o_d_assess_a_basic`

---

## Implementation Best Practices

### 1. Error Handling

```python
async def main(*args):
    try:
        task = await hooks.getTask()
        result = await solve_task(task)
        await hooks.submit(result)
    except Exception as e:
        hooks.log(f"Error: {str(e)}")
        # Submit partial result or error description
        await hooks.submit(f"Error encountered: {str(e)}")
```

### 2. Token Management

```python
async def check_token_usage():
    usage = await hooks.get_usage()
    used_ratio = usage.usage.tokens / usage.usageLimits.tokens
    
    if used_ratio > 0.9:
        hooks.log("Approaching token limit")
        # Implement token-saving strategies
        return True
    return False
```

### 3. State Persistence

```python
# Save state regularly for long-running agents
state = {
    "current_step": "analysis",
    "intermediate_results": results,
    "attempt_count": attempt_count
}
hooks.save_state(state)

# State is restored automatically on agent restart
```

### 4. Logging Best Practices

```python
# Use structured logging
hooks.log("🚀 Agent starting...")
hooks.log(f"📋 Task: {task.instructions[:100]}...")
hooks.log(f"✅ Solution generated: {len(solution)} chars")
hooks.log(f"⚠️ Warning: {warning_message}")
hooks.log(f"❌ Error: {error_message}")
```

### 5. Clean Output in Vivaria

```python
# Detect Vivaria environment
in_vivaria = os.getenv('RUN_ID') is not None

# Disable rich formatting if in Vivaria
if in_vivaria:
    # Use plain text output
    print("Simple message")
else:
    # Use rich formatting locally
    console.print("Fancy [bold]message[/bold]")
```

---

## Common Integration Patterns

### 1. Tool Adaptation

```python
class VivariaToolAdapter:
    def __init__(self, hooks):
        self.hooks = hooks
        
    async def run_bash(self, command: str) -> str:
        result = await self.hooks.action("bash", command=command)
        return result.stdout + result.stderr
        
    async def run_python(self, code: str) -> str:
        result = await self.hooks.action("python", code=code)
        return result.stdout + result.stderr
        
    async def score_solution(self) -> float:
        result = await self.hooks.score()
        return result.score
```

### 2. LLM Integration

```python
# Use Vivaria's middleman for LLM calls
from pyhooks.types import MiddlemanSettings

async def get_llm_response(prompt: str, model: str = "gpt-4o"):
    settings = MiddlemanSettings(
        model=model,
        temperature=0.1,
        max_tokens=4096
    )
    
    result = await hooks.middleman_generate(
        settings=settings,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return result.outputs[0].completion
```

### 3. File Operations

```python
# Working directory is typically /home/agent/solution
import os

def save_solution(content: str, filename: str = "solution.py"):
    solution_path = "/home/agent/solution"
    os.makedirs(solution_path, exist_ok=True)
    
    with open(f"{solution_path}/{filename}", "w") as f:
        f.write(content)
    
    hooks.log(f"Saved solution to {filename}")
```

---

## Troubleshooting

### Common Issues

1. **Agent submits too quickly**
   - Remove references to non-existent tools
   - Update prompts to be Vivaria-specific
   - Ensure proper task analysis before submission

2. **Import errors**
   - Check that all dependencies are in `requirements.txt`
   - Use absolute imports or adjust `sys.path`
   - Ensure agent code is properly structured

3. **Token limit exceeded**
   - Implement token tracking
   - Use more efficient prompting strategies
   - Consider model with higher limits (o1, o3)

4. **State not persisting**
   - Call `hooks.save_state()` regularly
   - Ensure state is JSON-serializable
   - Handle state restoration in agent initialization

5. **Actions failing**
   - Check action syntax: `await hooks.action("bash", command=cmd)`
   - Verify tool parameters match expected format
   - Handle action results properly

### Debugging Tips

```python
# Enable debug logging
hooks.log(f"DEBUG: Current state: {agent.state}")
hooks.log(f"DEBUG: Token usage: {usage.usage.tokens}")

# Inspect Vivaria environment
hooks.log(f"TASK_ID: {os.getenv('TASK_ID')}")
hooks.log(f"RUN_ID: {os.getenv('RUN_ID')}")
hooks.log(f"Working dir: {os.getcwd()}")

# Test actions independently
test_result = await hooks.action("bash", command="echo 'test'")
hooks.log(f"Test action result: {test_result}")
```

---

## Example: Minimal Working Agent

```python
#!/usr/bin/env python3
"""
Minimal Vivaria agent example
"""
from pyhooks import Hooks

hooks = Hooks()

async def main(*args):
    """Main agent function"""
    hooks.log("🚀 Agent starting...")
    
    # Get task
    task = await hooks.getTask()
    hooks.log(f"📋 Task: {task.instructions}")
    
    # Simple solution approach
    analysis_code = f'''
# Analyze the task
task = """{task.instructions}"""
print("Task analysis:")
print(f"Task length: {{len(task)}} characters")
print("Key requirements identified")
'''
    
    # Run analysis
    result = await hooks.action("python", code=analysis_code)
    hooks.log(f"Analysis result: {result.stdout}")
    
    # Generate simple solution
    solution = "# Simple solution placeholder\nprint('Solution implemented')"
    
    # Save solution file
    save_code = f'''
with open("/home/agent/solution/solution.py", "w") as f:
    f.write("""{solution}""")
print("Solution saved")
'''
    
    await hooks.action("python", code=save_code)
    
    # Submit
    await hooks.submit("Solution implemented and saved")
    hooks.log("✅ Agent completed")

if __name__ == "__main__":
    hooks.main(main)
```

---

## References

- [Vivaria Documentation](https://vivaria.metr.org)
- [pyhooks Repository](https://github.com/METR/pyhooks)
- [modular-public Agent](https://github.com/poking-agents/modular-public)
- [Vivaria Agent Creation Tutorial](https://vivaria.metr.org/tutorials/create-agent)

---

This guide should help your team understand the patterns and requirements for creating effective Vivaria agents, whether building from scratch or adapting existing agent frameworks.
