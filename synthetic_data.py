import pandas as pd
import random

def generate_synthetic_dataset(num_agents=5, num_samples=20):
    data = []
    for agent in range(num_agents):
        for i in range(num_samples):
            instr = f"Task {i}: Summarize topic {random.randint(1,10)}"
            resp = f"This is a response to {instr}. Agent {agent} provides details."
            data.append({"agent": f"Agent_{agent}", "instruction": instr, "response": resp})
    return pd.DataFrame(data)
