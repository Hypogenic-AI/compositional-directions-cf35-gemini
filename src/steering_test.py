import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import os
import json

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

def get_logits(model, act_tensor):
    with torch.no_grad():
        norm_act = model.model.norm(act_tensor)
        logits = model.lm_head(norm_act)
    return logits[0, 0, :]

def main():
    model_name = "Qwen/Qwen2.5-1.5B"
    model, tokenizer = get_model(model_name)
    
    # 1. Source Directions
    print("Extracting directions...")
    # Color: Red vs Blue
    act_apple_red = get_last_act(model, tokenizer, "The apple is red")
    act_apple_blue = get_last_act(model, tokenizer, "The apple is blue")
    g_color = (act_apple_red - act_apple_blue)
    g_color = g_color / torch.norm(g_color)
    
    # Truth: True vs False
    act_fact_true = get_last_act(model, tokenizer, "Paris is in France. True")
    act_fact_false = get_last_act(model, tokenizer, "Paris is in Germany. False")
    g_truth = (act_fact_true - act_fact_false)
    g_truth = g_truth / torch.norm(g_truth)
    
    # 2. Target Prompts
    target_color = "The truck is "
    target_truth = "The statement 'London is in England' is "
    
    red_id = tokenizer.encode(" red")[0]
    blue_id = tokenizer.encode(" blue")[0]
    true_id = tokenizer.encode(" True")[0]
    false_id = tokenizer.encode(" False")[0]
    
    results = []
    
    alphas = [0, 5, 10, 20]
    
    print("\n--- Steering 'The truck is ' with Color Direction ---")
    base_act = get_last_act(model, tokenizer, target_color)
    for a in alphas:
        steered_act = base_act + a * g_color
        logits = get_logits(model, steered_act)
        p_red = torch.softmax(logits, dim=-1)[red_id].item()
        p_blue = torch.softmax(logits, dim=-1)[blue_id].item()
        print(f"Alpha {a:2d}: P(red)={p_red:.4f}, P(blue)={p_blue:.4f}, Diff={p_red-p_blue:.4f}")

    print("\n--- Steering 'The truck is ' with Truth Direction (Unrelated) ---")
    for a in alphas:
        steered_act = base_act + a * g_truth
        logits = get_logits(model, steered_act)
        p_red = torch.softmax(logits, dim=-1)[red_id].item()
        p_blue = torch.softmax(logits, dim=-1)[blue_id].item()
        print(f"Alpha {a:2d}: P(red)={p_red:.4f}, P(blue)={p_blue:.4f}, Diff={p_red-p_blue:.4f}")

    print("\n--- Steering Truth Prompt with Truth Direction (Related) ---")
    base_act_truth = get_last_act(model, tokenizer, target_truth)
    for a in alphas:
        steered_act = base_act_truth + a * g_truth
        logits = get_logits(model, steered_act)
        p_true = torch.softmax(logits, dim=-1)[true_id].item()
        p_false = torch.softmax(logits, dim=-1)[false_id].item()
        print(f"Alpha {a:2d}: P(True)={p_true:.4f}, P(False)={p_false:.4f}, Diff={p_true-p_false:.4f}")

    print("\n--- Steering Truth Prompt with Color Direction (Unrelated) ---")
    for a in alphas:
        steered_act = base_act_truth + a * g_color
        logits = get_logits(model, steered_act)
        p_true = torch.softmax(logits, dim=-1)[true_id].item()
        p_false = torch.softmax(logits, dim=-1)[false_id].item()
        print(f"Alpha {a:2d}: P(True)={p_true:.4f}, P(False)={p_false:.4f}, Diff={p_true-p_false:.4f}")

if __name__ == "__main__":
    main()
