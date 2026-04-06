import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

def get_model(model_name="Qwen/Qwen2.5-1.5B"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float32, # Use float32 for stability
        output_hidden_states=True
    )
    return model, tokenizer

def get_last_act(model, tokenizer, text, layer_idx=-1):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    hidden_states = outputs.hidden_states[layer_idx]
    return hidden_states[0, -1, :].cpu().numpy()

def main():
    model_name = "Qwen/Qwen2.5-1.5B"
    model, tokenizer = get_model(model_name)
    
    concepts = [("red", "blue"), ("True", "False")]
    
    for c1, c2 in concepts:
        v1 = get_last_act(model, tokenizer, c1)
        v2 = get_last_act(model, tokenizer, c2)
        d = v1 - v2
        norm = np.linalg.norm(d)
        print(f"Concept {c1}-{c2}: Norm={norm:.4f}, Mean={np.mean(d):.4f}, Max={np.max(np.abs(d)):.4f}")

if __name__ == "__main__":
    main()
