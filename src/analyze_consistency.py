import json
import numpy as np
from scipy.spatial.distance import cosine

def load_data(path="results/activations_v3.json"):
    with open(path, "r") as f:
        return json.load(f)

def get_cos_sim(v1, v2):
    return 1 - cosine(v1, v2)

def main():
    data = load_data()
    
    summary = []
    for layer in sorted(data.keys(), key=lambda x: int(x) if x != "-1" else 99):
        layer_data = data[layer]
        bases = np.array(layer_data["bases"]) # (N_obj, D)
        
        # For each concept (red, blue, green, prime number, etc.)
        concepts = ["red", "blue", "green", "prime number", "democracy", "truth"]
        
        concept_stats = {}
        for c in concepts:
            composed = np.array(layer_data[f"{c}_composed"])
            # Directions: gamma_i = composed_i - base_i
            diffs = composed - bases
            
            # 1. Consistency: How similar are the directions across objects?
            # Pairwise cosine similarity of diffs
            sims = []
            for i in range(len(diffs)):
                for j in range(i + 1, len(diffs)):
                    sims.append(get_cos_sim(diffs[i], diffs[j]))
            
            # 2. Orthogonality to base: Are we moving in a new direction?
            # cos(diff_i, base_i)
            ortho_sims = [get_cos_sim(diffs[i], bases[i]) for i in range(len(diffs))]
            
            concept_stats[c] = {
                "consistency": np.mean(sims),
                "ortho": np.mean(ortho_sims)
            }
            
        rel_cons = np.mean([concept_stats[c]["consistency"] for c in ["red", "blue", "green"]])
        unrel_cons = np.mean([concept_stats[c]["consistency"] for c in ["prime number", "democracy", "truth"]])
        
        summary.append({
            "layer": layer,
            "related_consistency": rel_cons,
            "unrelated_consistency": unrel_cons,
            "gap": rel_cons - unrel_cons
        })
        
    print(f"{'Layer':<10} | {'Rel Cons':<10} | {'Unrel Cons':<10} | {'Gap':<10}")
    print("-" * 50)
    for s in summary:
        print(f"{s['layer']:<10} | {s['related_consistency']:<10.4f} | {s['unrelated_consistency']:<10.4f} | {s['gap']:<10.4f}")

    with open("results/consistency_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
