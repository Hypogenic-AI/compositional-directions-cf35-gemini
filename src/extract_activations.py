import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import os
import json
from tqdm import tqdm

def get_model(model_name="Qwen/Qwen2.5-1.5B"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        output_hidden_states=True
    )
    return model, tokenizer

def get_activations(model, tokenizer, texts, layer_idx=-1):
    """
    Get activations for the last token of each text in the batch.
    """
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # hidden_states is a tuple of (layer_count + 1) tensors
    # each tensor is (batch_size, seq_len, hidden_size)
    hidden_states = outputs.hidden_states[layer_idx]
    
    # Get last token position for each sequence
    attention_mask = inputs["attention_mask"]
    last_token_indices = attention_mask.sum(dim=1) - 1
    
    batch_size = hidden_states.shape[0]
    activations = hidden_states[torch.arange(batch_size), last_token_indices, :]
    
    return activations.cpu().numpy()

def main():
    model_name = "Qwen/Qwen2.5-1.5B"
    model, tokenizer = get_model(model_name)
    
    # Define test cases
    # We use neutral prompts to better isolate the concept
    neutral_prompt = "A "
    
    test_cases = {
        "neutral": ["A"],
        "colors": ["red", "blue", "green", "yellow", "black", "white"],
        "objects": ["car", "apple", "truck", "ball", "house", "bird"],
        "abstract": ["prime number", "democracy", "truth", "justice", "logic"],
        "related_compositions": [
            "red car", "blue car", "green car", 
            "red apple", "green apple",
            "yellow ball", "black bird"
        ],
        "unrelated_compositions": [
            "red prime number", "blue democracy", "green truth",
            "car justice", "apple logic", "yellow prime number"
        ]
    }
    
    # Prepend neutral prompt
    for cat in test_cases:
        if cat != "neutral":
            test_cases[cat] = [neutral_prompt + t for t in test_cases[cat]]
    
    results = {}
    layers = [5, 10, 15, 20, -1] # Extract from multiple depths
    
    for layer in layers:
        layer_results = {}
        for category, texts in test_cases.items():
            print(f"Extracting {category} from layer {layer}...")
            acts = get_activations(model, tokenizer, texts, layer_idx=layer)
            layer_results[category] = {
                "texts": texts,
                "activations": acts.tolist()
            }
        results[str(layer)] = layer_results

    os.makedirs("results", exist_ok=True)
    with open("results/activations.json", "w") as f:
        json.dump(results, f)
    
    print("Done! Activations saved to results/activations.json")

if __name__ == "__main__":
    main()
