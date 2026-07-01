import os

agents = ['whisper_agent', 'embedding_agent', 'evaluation_agent']
template = """\
\"\"\"
{name} — AI Video Director
==================================
LangGraph Node
\"\"\"
from memory.state import VideoState

def {name}_node(state: VideoState) -> VideoState:
    print("Running {name}...")
    state['current_agent'] = '{name}'
    if 'logs' not in state:
        state['logs'] = []
    state['logs'].append(f"{name} completed.")
    return state
"""

for a in agents:
    with open(f'agents/{a}.py', 'w') as f:
        f.write(template.format(name=a))
print('Scaffolded remaining agents')
