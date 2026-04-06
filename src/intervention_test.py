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

def get_logits_from_act(model, act):
    # act is (D,)
    # Qwen-2.5-1.5B has a final LayerNorm before the lm_head
    # We should apply it if we took activations before it.
    # But get_activations_at_pos used hidden_states[layer_idx], 
    # and layer_idx=-1 is the output of the last block.
    # The lm_head usually takes the output of the final norm.
    
    # For simplicity, we use the model's head.
    # We need to apply the final norm!
    
    with torch.no_grad():
        # norm is model.model.norm
        act_tensor = torch.tensor(act, dtype=torch.float16).to(model.device).reshape(1, 1, -1)
        # Apply the rest of the model after the last block
        # Actually, if we are at layer -1, we just need the final norm and head.
        norm_act = model.model.norm(act_tensor)
        logits = model.lm_head(norm_act)
    
    return logits[0, 0, :].cpu().numpy()

def main():
    model_name = "Qwen/Qwen2.5-1.5B"
    model, tokenizer = get_model(model_name)
    
    with open("results/activations_v3.json", "r") as f:
        data = json.load(f)["-1"] # Use last layer
    
    bases = np.array(data["bases"]) # ["car", "apple", "truck", "ball", "house", "bird"]
    red_composed = np.array(data["red_composed"])
    blue_composed = np.array(data["blue_composed"])
    
    # Target object: car (index 0)
    # Source object for direction: apple (index 1)
    
    # 1. Related Intervention: Turn blue car into red car using apple's red-direction
    gamma_red_apple = np.array(data["red_composed"][1]) - np.array(data["bases"][1])
    
    blue_car_act = np.array(data["blue_composed"][0])
    intervened_act = blue_car_act + gamma_red_apple
    
    # 2. Unrelated Intervention: Turn blue car into "truth" car
    gamma_truth_apple = np.array(data["truth_composed"][1]) - np.array(data["bases"][1])
    intervened_unrelated_act = blue_car_act + gamma_truth_apple
    
    # Check logits
    def check_top_k(act, name, k=5):
        logits = get_logits_from_act(model, act)
        top_indices = np.argsort(logits)[::-1][:k]
        top_tokens = [tokenizer.decode(i) for i in top_indices]
        print(f"Top {k} for {name}: {top_tokens}")
        return top_tokens

    print("--- Related ---")
    check_top_k(blue_car_act, "Blue Car (Original)")
    check_top_k(red_composed[0], "Red Car (Actual)")
    check_top_k(intervened_act, "Intervened (Blue Car + Red-from-Apple)")
    
    print("\n--- Unrelated ---")
    check_top_k(data["truth_composed"][0], "Truth Car (Actual)")
    check_top_k(intervened_unrelated_act, "Intervened (Blue Car + Truth-from-Apple)")

if __name__ == "__main__":
    main()
