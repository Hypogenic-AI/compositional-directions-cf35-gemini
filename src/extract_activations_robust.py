import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import os
import json
import pandas as pd

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
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    hidden_states = outputs.hidden_states[layer_idx]
    attention_mask = inputs["attention_mask"]
    last_token_indices = attention_mask.sum(dim=1) - 1
    batch_size = hidden_states.shape[0]
    activations = hidden_states[torch.arange(batch_size), last_token_indices, :]
    return activations.cpu().numpy()

def load_park_pairs(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    return [line.strip().split('\t') for line in lines if line.strip()]

def main():
    model_name = "Qwen/Qwen2.5-1.5B"
    model, tokenizer = get_model(model_name)
    
    # 1. Gender Directions
    # Royalty
    royalty_pairs = [["king", "queen"], ["prince", "princess"], ["lord", "lady"]]
    # Family
    family_pairs = [["father", "mother"], ["son", "daughter"], ["uncle", "aunt"], ["brother", "sister"]]
    
    # 2. Truth Directions (from datasets/truthful_qa.csv or similar)
    # We'll use simple facts
    animal_facts = [
        ["A cat is a mammal", "A cat is a bird"],
        ["A dog is a mammal", "A dog is a fish"],
        ["A robin is a bird", "A robin is a mammal"],
        ["A shark is a fish", "A shark is a bird"]
    ]
    city_facts = [
        ["Paris is in France", "Paris is in Germany"],
        ["London is in England", "London is in France"],
        ["Berlin is in Germany", "Berlin is in Italy"],
        ["Rome is in Italy", "Rome is in Spain"]
    ]

    results = {}
    
    for layer in [10, 15, 20, -1]:
        print(f"Processing layer {layer}...")
        layer_results = {}
        
        # Royalty Gender
        royalty_acts = []
        for p in royalty_pairs:
            royalty_acts.append(get_activations(model, tokenizer, p, layer_idx=layer))
        layer_results["royalty_gender"] = np.array(royalty_acts).tolist() # (N, 2, D)
        
        # Family Gender
        family_acts = []
        for p in family_pairs:
            family_acts.append(get_activations(model, tokenizer, p, layer_idx=layer))
        layer_results["family_gender"] = np.array(family_acts).tolist()
        
        # Animal Truth
        animal_acts = []
        for p in animal_facts:
            animal_acts.append(get_activations(model, tokenizer, p, layer_idx=layer))
        layer_results["animal_truth"] = np.array(animal_acts).tolist()
        
        # City Truth
        city_acts = []
        for p in city_facts:
            city_acts.append(get_activations(model, tokenizer, p, layer_idx=layer))
        layer_results["city_truth"] = np.array(city_acts).tolist()
        
        results[str(layer)] = layer_results

    os.makedirs("results", exist_ok=True)
    with open("results/activations_robust.json", "w") as f:
        json.dump(results, f)
    
    print("Done!")

if __name__ == "__main__":
    main()
