import os

agents = [
    'input_agent', 'preprocess_agent', 'scene_agent', 'metadata_agent',
    'rag_agent', 'planner_agent', 'caption_agent', 'editing_agent',
    'reviewer_agent', 'export_agent'
]

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

tools = ['whisper', 'ffmpeg', 'opencv', 'embeddings', 'reranker']
os.makedirs('tools', exist_ok=True)
for t in tools:
    with open(f'tools/{t}.py', 'w') as f:
        f.write(f'# Tool: {t}\n')

print("Scaffolded agents and tools")
