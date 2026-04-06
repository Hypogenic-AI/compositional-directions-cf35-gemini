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
    return hidden_states[0, -1, :].reshape(1, -1).cpu().numpy()

def main():
    model_name = "Qwen/Qwen2.5-1.5B"
    model, tokenizer = get_model(model_name)
    
    # 1. Gather a set of directions to compute Covariance
    print("Extracting a set of directions...")
    
    # We'll use the word pairs from the repo if possible, but let's just use some manual ones for speed.
    concepts = [
        ("red", "blue"), ("green", "yellow"), ("black", "white"),
        ("True", "False"), ("Yes", "No"), ("Correct", "Incorrect"),
        ("man", "woman"), ("king", "queen"), ("father", "mother"),
        ("up", "down"), ("left", "right"), ("large", "small")
    ]
    
    dirs = []
    for c1, c2 in concepts:
        v1 = get_last_act(model, tokenizer, c1)
        v2 = get_last_act(model, tokenizer, c2)
        d = v1 - v2
        dirs.append(d[0])
    
    dirs = np.array(dirs) # (N, D)
    
    # Compute Covariance and its (pseudo)inverse
    cov = np.cov(dirs.T)
    # Since D is large (1536 or 2048), cov is huge.
    # We only need the inner products.
    # Actually, we can use the empirical directions themselves.
    
    # For simplicity, let's just use the raw cosine similarity first, 
    # and then the "Whitened" similarity (equivalent to using Cov^-1).
    
    def whitened_sim(v1, v2, directions):
        # We project onto the subspace of these directions
        # And then whiten
        # Actually, let's just do a simpler version: 
        # Orthogonality of Related vs Unrelated groups
        return 
    
    # Let's just compare the average cosine similarity within and between groups.
    groups = {
        "color": dirs[0:3],
        "truth": dirs[3:6],
        "gender": dirs[6:9],
        "spatial": dirs[9:12]
    }
    
    keys = list(groups.keys())
    print(f"\n{'Group 1':<10} | {'Group 2':<10} | {'Avg Cosine Sim':<15}")
    print("-" * 45)
    
    for i in range(len(keys)):
        for j in range(i, len(keys)):
            g1 = groups[keys[i]]
            g2 = groups[keys[j]]
            
            sims = []
            for v1 in g1:
                for v2 in g2:
                    if np.array_equal(v1, v2): continue
                    s = 1 - cosine(v1, v2)
                    sims.append(s)
            
            print(f"{keys[i]:<10} | {keys[j]:<10} | {np.mean(sims):<15.4f}")

if __name__ == "__main__":
    from scipy.spatial.distance import cosine
    main()
