import requests
import os

papers = {
    "park2023_linear_representation_hypothesis.pdf": "https://arxiv.org/pdf/2311.04017.pdf",
    "marks2023_geometry_of_truth.pdf": "https://arxiv.org/pdf/2310.16825.pdf",
    "azizian2025_geometries_of_truth_orthogonal.pdf": "https://arxiv.org/pdf/2501.14656.pdf",
    "khandelwal2025_composing_functions.pdf": "https://arxiv.org/pdf/2410.15372.pdf",
    "polysae2026_compositional_sae.pdf": "https://arxiv.org/pdf/2501.07166.pdf",
    "lippl2025_algorithmic_primitives.pdf": "https://arxiv.org/pdf/2502.16431.pdf",
    "wattenberg2024_relational_composition.pdf": "https://arxiv.org/pdf/2407.12154.pdf"
}

os.makedirs("papers", exist_ok=True)

for filename, url in papers.items():
    print(f"Downloading {filename}...")
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        with open(os.path.join("papers", filename), "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Successfully downloaded {filename}")
    except Exception as e:
        print(f"Failed to download {filename}: {e}")
