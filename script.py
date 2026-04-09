import json

with open("slm_colab.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

nb.get("metadata", {}).pop("widgets", None)

with open("slm_colab_clean.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)