from src.llm_api import call_llm
from src.parser import parse_answer, parse_confidence
from src.prompts import build_prompt, build_sequential_followup


SINGLE_INPUT_MODE = "single_input"
SEQUENTIAL_INPUT_MODE = "sequential_input"
INPUT_MODES = [SINGLE_INPUT_MODE, SEQUENTIAL_INPUT_MODE]


def run_initial_pass(item):
    initial_messages = build_prompt(item, opinions=[])
    initial_text = call_llm(initial_messages)
    return {
        "messages": initial_messages,
        "text": initial_text,
        "prediction": parse_answer(initial_text),
        "confidence": parse_confidence(initial_text),
    }


def run_attack_pass(item, opinions, attack_level, initial_result, input_mode):
    if input_mode == SINGLE_INPUT_MODE:
        attack_messages = build_prompt(item, opinions=opinions, attack_level=attack_level)
        attack_text = call_llm(attack_messages)
        return {
            "input_mode": input_mode,
            "messages": attack_messages,
            "text": attack_text,
            "prediction": parse_answer(attack_text),
            "confidence": parse_confidence(attack_text),
            "step_outputs": [],
        }

    if input_mode == SEQUENTIAL_INPUT_MODE:
        messages = list(initial_result["messages"])
        messages.append({"role": "assistant", "content": initial_result["text"]})

        step_outputs = []
        final_text = ""
        for agent_index, opinion in enumerate(opinions, start=1):
            followup = build_sequential_followup(opinion, agent_index, attack_level=attack_level)
            messages.append({"role": "user", "content": followup})
            step_text = call_llm(messages)
            messages.append({"role": "assistant", "content": step_text})
            step_outputs.append(
                {
                    "agent_index": agent_index,
                    "opinion": opinion,
                    "output": step_text,
                    "prediction": parse_answer(step_text),
                    "confidence": parse_confidence(step_text),
                }
            )
            final_text = step_text

        return {
            "input_mode": input_mode,
            "messages": messages,
            "text": final_text,
            "prediction": parse_answer(final_text),
            "confidence": parse_confidence(final_text),
            "step_outputs": step_outputs,
        }

    raise ValueError(f"Unsupported input_mode: {input_mode}")
