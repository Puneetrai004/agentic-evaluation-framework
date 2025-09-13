import pandas as pd
import textstat
import language_tool_python

tool = language_tool_python.LanguageTool('en-US')

def evaluate_responses(df, use_llm_judge=False):
    scores = []
    for _, row in df.iterrows():
        response = row["response"]

        # Rule-based metrics
        grammar_matches = len(tool.check(response))
        readability = textstat.flesch_reading_ease(response)

        # Simple scoring
        instruction_follow = 1 if row["instruction"].lower() in response.lower() else 0
        coherence = 1 if readability > 40 else 0
        grammar_score = max(0, 1 - grammar_matches/10)

        final_score = (instruction_follow + coherence + grammar_score) / 3

        # Optional LLM judge (stubbed for now, can hook Hugging Face API later)
        if use_llm_judge:
            final_score = (final_score + 0.8) / 2  # Example: trust LLM judge

        scores.append({
            "agent": row["agent"],
            "instruction": row["instruction"],
            "response": response,
            "score_instruction": instruction_follow,
            "score_coherence": coherence,
            "score_grammar": grammar_score,
            "final_score": final_score
        })

    return pd.DataFrame(scores)
