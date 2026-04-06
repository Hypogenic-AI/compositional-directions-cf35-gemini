import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import json
import os

def get_model(model_name="Qwen/Qwen2.5-1.5B"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        output_hidden_states=True
    )
    return model, tokenizer

def get_last_act(model, tokenizer, text, layer_idx=-1):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    hidden_states = outputs.hidden_states[layer_idx]
    return hidden_states[0, -1, :].reshape(1, 1, -1)

def get_prob(model, act_tensor, token_id):
    with torch.no_grad():
        norm_act = model.model.norm(act_tensor)
        logits = model.lm_head(norm_act)
    probs = torch.softmax(logits[0, 0, :], dim=-1)
    return probs[token_id].item()

def main():
    model_name = "Qwen/Qwen2.5-1.5B"
    model, tokenizer = get_model(model_name)
    
    # 1. Directions
    # Color
    g_color = get_last_act(model, tokenizer, "The apple is red") - get_last_act(model, tokenizer, "The apple is blue")
    g_color = g_color / torch.norm(g_color)
    
    # Truth
    g_truth = get_last_act(model, tokenizer, "True") - get_last_act(model, tokenizer, "False")
    g_truth = g_truth / torch.norm(g_truth)
    
    # 2. Tasks
    tasks = [
        {
            "name": "Truck Color (Target: red)",
            "prompt": "The truck is painted",
            "target_token": " red",
            "related_direction": g_color,
            "unrelated_direction": g_truth
        },
        {
            "name": "Fact Truth (Target: True)",
            "prompt": "The capital of France is Paris. This statement is",
            "target_token": " True",
            "related_direction": g_truth,
            "unrelated_direction": g_color
        }
    ]
    
    results = []
    alphas = [0, 5, 10, 20]
    
    for task in tasks:
        task_res = {"name": task["name"], "related": [], "unrelated": []}
        token_id = tokenizer.encode(task["target_token"])[0]
        base_act = get_last_act(model, tokenizer, task["prompt"])
        
        print(f"\nTask: {task['name']}")
        for a in alphas:
            p_rel = get_prob(model, base_act + a * task["related_direction"], token_id)
            p_unrel = get_prob(model, base_act + a * task["unrelated_direction"], token_id)
            task_res["related"].append(p_rel)
            task_res["unrelated"].append(p_unrel)
            print(f"Alpha {a:2d} | Related P: {p_rel:.4f} | Unrelated P: {p_unrel:.4f}")
        results.append(task_res)
        
    os.makedirs("results", exist_ok=True)
    with open("results/steering_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
