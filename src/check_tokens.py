import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

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

def get_top_k(model, tokenizer, act_tensor, k=10):
    with torch.no_grad():
        norm_act = model.model.norm(act_tensor)
        logits = model.lm_head(norm_act)
    probs = torch.softmax(logits[0, 0, :], dim=-1)
    top_vals, top_indices = torch.topk(probs, k=k)
    return [(tokenizer.decode(i), val.item()) for i, val in zip(top_indices, top_vals)]

def main():
    model_name = "Qwen/Qwen2.5-1.5B"
    model, tokenizer = get_model(model_name)
    
    # Color Direction
    act_apple_red = get_last_act(model, tokenizer, "The apple is red")
    act_apple_blue = get_last_act(model, tokenizer, "The apple is blue")
    g_color = (act_apple_red - act_apple_blue)
    g_color = g_color / torch.norm(g_color)
    
    target_color = "The truck is painted" # Changed to painted to narrow down
    print(f"\nTarget: {target_color}")
    base_act = get_last_act(model, tokenizer, target_color)
    
    print("Baseline Top 5:")
    print(get_top_k(model, tokenizer, base_act, k=5))
    
    print("Steered (Alpha=20) Top 10:")
    steered_act = base_act + 20 * g_color
    print(get_top_k(model, tokenizer, steered_act, k=10))

if __name__ == "__main__":
    main()
