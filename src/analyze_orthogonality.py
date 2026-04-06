import json
import numpy as np
from scipy.spatial.distance import cosine

def load_data(path="results/activations_robust.json"):
    with open(path, "r") as f:
        return json.load(f)

def get_direction(acts_list):
    # acts_list is (N, 2, D)
    # diffs = acts[:, 1, :] - acts[:, 0, :]
    acts = np.array(acts_list)
    diffs = acts[:, 1, :] - acts[:, 0, :]
    mean_diff = np.mean(diffs, axis=0)
    return mean_diff / np.linalg.norm(mean_diff)

def get_cos_sim(v1, v2):
    return 1 - cosine(v1, v2)

def main():
    data = load_data()
    
    summary = []
    for layer in sorted(data.keys(), key=lambda x: int(x) if x != "-1" else 99):
        layer_data = data[layer]
        
        g_royalty = get_direction(layer_data["royalty_gender"])
        g_family = get_direction(layer_data["family_gender"])
        g_animal = get_direction(layer_data["animal_truth"])
        g_city = get_direction(layer_data["city_truth"])
        
        sim_gender = get_cos_sim(g_royalty, g_family)
        sim_truth = get_cos_sim(g_animal, g_city)
        
        summary.append({
            "layer": layer,
            "gender_sim": sim_gender,
            "truth_sim": sim_truth,
            "gap": sim_gender - sim_truth
        })
        
    print(f"{'Layer':<10} | {'Gender Sim':<12} | {'Truth Sim':<12} | {'Gap':<10}")
    print("-" * 55)
    for s in summary:
        print(f"{s['layer']:<10} | {s['gender_sim']:<12.4f} | {s['truth_sim']:<12.4f} | {s['gap']:<10.4f}")

    with open("results/orthogonality_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
