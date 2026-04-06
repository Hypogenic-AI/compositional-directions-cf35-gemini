import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cosine

def load_data(path="results/activations.json"):
    with open(path, "r") as f:
        return json.load(f)

def get_cos_sim(v1, v2):
    return 1 - cosine(v1, v2)

def analyze_layer(layer_data):
    neutral_act = np.mean(layer_data["neutral"]["activations"], axis=0)
    
    # Compute directions
    directions = {}
    for cat in ["colors", "objects", "abstract"]:
        for text, act in zip(layer_data[cat]["texts"], layer_data[cat]["activations"]):
            # remove "A " prefix
            name = text[2:]
            directions[name] = np.array(act) - neutral_act
            
    # Analyze compositions
    results = {"related": [], "unrelated": []}
    
    # Related compositions
    for text, act in zip(layer_data["related_compositions"]["texts"], layer_data["related_compositions"]["activations"]):
        name = text[2:] # "red car"
        parts = name.split()
        if len(parts) == 2:
            p1, p2 = parts
            if p1 in directions and p2 in directions:
                v_sum = directions[p1] + directions[p2]
                v_target = np.array(act) - neutral_act
                sim = get_cos_sim(v_sum, v_target)
                results["related"].append({"name": name, "sim": sim})
                
    # Unrelated compositions
    for text, act in zip(layer_data["unrelated_compositions"]["texts"], layer_data["unrelated_compositions"]["activations"]):
        name = text[2:]
        parts = name.split()
        # Handle cases like "prime number" which is 2 words but treated as one concept
        # Actually our "abstract" list has "prime number" as one entry.
        # But in composition it might be "red prime number" (3 words).
        if name.startswith("red prime number"):
            p1 = "red"
            p2 = "prime number"
        elif name.startswith("yellow prime number"):
            p1 = "yellow"
            p2 = "prime number"
        else:
            p1, p2 = parts[0], " ".join(parts[1:])
            
        if p1 in directions and p2 in directions:
            v_sum = directions[p1] + directions[p2]
            v_target = np.array(act) - neutral_act
            sim = get_cos_sim(v_sum, v_target)
            results["unrelated"].append({"name": name, "sim": sim})
            
    return results

def main():
    data = load_data()
    all_results = {}
    
    for layer in data:
        all_results[layer] = analyze_layer(data[layer])
        
    # Summarize results
    summary = []
    for layer in sorted(all_results.keys(), key=lambda x: int(x) if x != "-1" else 99):
        rel_sims = [r["sim"] for r in all_results[layer]["related"]]
        unrel_sims = [r["sim"] for r in all_results[layer]["unrelated"]]
        
        summary.append({
            "layer": layer,
            "related_mean": np.mean(rel_sims),
            "unrelated_mean": np.mean(unrel_sims),
            "diff": np.mean(rel_sims) - np.mean(unrel_sims)
        })
        
    print(f"{'Layer':<10} | {'Related':<10} | {'Unrelated':<10} | {'Diff':<10}")
    print("-" * 50)
    for s in summary:
        print(f"{s['layer']:<10} | {s['related_mean']:<10.4f} | {s['unrelated_mean']:<10.4f} | {s['diff']:<10.4f}")

    # Visualization
    layers = [s["layer"] for s in summary]
    rel = [s["related_mean"] for s in summary]
    unrel = [s["unrelated_mean"] for s in summary]
    
    plt.figure(figsize=(10, 6))
    plt.plot(layers, rel, label="Related", marker='o')
    plt.plot(layers, unrel, label="Unrelated", marker='s')
    plt.xlabel("Layer")
    plt.ylabel("Cosine Similarity: (v1+v2) vs v(1+2)")
    plt.title("Compositionality of Linear Directions (Qwen-1.5B)")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/compositionality_plot.png")
    
    with open("results/analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
