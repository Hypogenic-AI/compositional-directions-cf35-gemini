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

def get_activations_at_pos(model, tokenizer, texts, target_pos, layer_idx=-1):
    """
    Get activations for a token at a specific position.
    We pad the beginning to ensure the target token is at target_pos.
    """
    # This is tricky with BPE. 
    # Better: Use a fixed prefix and ensure the total token count is constant.
    # "A A A A A A A A A [text]"
    prefix = "The object is " # 3 tokens in some tokenizers
    
    activations = []
    for text in texts:
        full_text = prefix + text
        inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
        # Find position of the last token of 'text' in full_text
        with torch.no_grad():
            outputs = model(**inputs)
        
        hidden_states = outputs.hidden_states[layer_idx]
        # We take the LAST token activation for the composed phrase
        # E.g. for "red car", it's "car". For "car", it's "car".
        act = hidden_states[0, -1, :].cpu().numpy()
        activations.append(act)
        
    return np.array(activations)

def main():
    model_name = "Qwen/Qwen2.5-1.5B"
    model, tokenizer = get_model(model_name)
    
    # We want to compare v(red car) vs v(car) + gamma(red)
    # How to get gamma(red)?
    # gamma(red) = v(red car) - v(car)? No, that's what we want to test.
    # gamma(red) = v(red apple) - v(apple)?
    
    # Let's get activations for:
    # 1. "car", "apple", "truck" (Base objects)
    # 2. "red car", "red apple", "red truck" (Red versions)
    # 3. "blue car", "blue apple", "blue truck" (Blue versions)
    
    # Then gamma(red)_car = v(red car) - v(car)
    # gamma(red)_apple = v(red apple) - v(apple)
    # If directions are compositional/universal, then gamma(red)_car approx gamma(red)_apple.
    
    test_objects = ["car", "apple", "truck", "ball", "house", "bird"]
    colors = ["red", "blue", "green"]
    abstracts = ["prime number", "democracy", "truth"]

    results = {}
    
    for layer in [15, 20, -1]:
        print(f"Processing layer {layer}...")
        layer_results = {}
        
        # Base objects
        layer_results["bases"] = get_activations_at_pos(model, tokenizer, test_objects, None, layer_idx=layer).tolist()
        
        # Color Compositions
        for color in colors:
            composed = [f"{color} {obj}" for obj in test_objects]
            layer_results[f"{color}_composed"] = get_activations_at_pos(model, tokenizer, composed, None, layer_idx=layer).tolist()
            
        # Abstract Compositions (Unrelated)
        for abs_concept in abstracts:
            composed = [f"{abs_concept} {obj}" for obj in test_objects]
            layer_results[f"{abs_concept}_composed"] = get_activations_at_pos(model, tokenizer, composed, None, layer_idx=layer).tolist()
            
        results[str(layer)] = layer_results

    os.makedirs("results", exist_ok=True)
    with open("results/activations_v3.json", "w") as f:
        json.dump(results, f)
    
    print("Done!")

if __name__ == "__main__":
    main()
