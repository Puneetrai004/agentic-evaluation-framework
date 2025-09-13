import gradio as gr
import pandas as pd
from evaluator import evaluate_responses
from synthetic_data import generate_synthetic_dataset

# Demo synthetic dataset
df = generate_synthetic_dataset(num_agents=10, num_samples=50)

def run_evaluation(use_llm_judge=False):
    results = evaluate_responses(df, use_llm_judge=use_llm_judge)
    leaderboard = results.groupby("agent")["final_score"].mean().reset_index()
    leaderboard = leaderboard.sort_values("final_score", ascending=False)
    return results, leaderboard

with gr.Blocks(title="Agentic Evaluation Framework") as demo:
    gr.Markdown("# 🤖 Agentic Evaluation Framework")
    gr.Markdown("Automatically evaluate AI agents across multiple dimensions.")

    with gr.Tab("Synthetic Data Preview"):
        gr.DataFrame(df, label="Generated Dataset", interactive=False)

    with gr.Tab("Run Evaluation"):
        use_llm = gr.Checkbox(label="Use LLM Judge (Optional)", value=False)
        run_button = gr.Button("Run Evaluation")
        results_output = gr.DataFrame(label="Evaluation Results")
        leaderboard_output = gr.DataFrame(label="Leaderboard")

        run_button.click(
            fn=run_evaluation,
            inputs=[use_llm],
            outputs=[results_output, leaderboard_output]
        )

demo.launch()
