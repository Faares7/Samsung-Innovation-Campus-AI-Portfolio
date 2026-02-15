import nbmerge

# List your notebooks in the order you want them merged
merged = nbmerge.merge([
    "heart_failure(1).ipynb",
    "Tree-Ensembles.ipynb",
    "Capstone2(NB,KNN,LR,SVM).ipynb"
])

# Save to a new notebook
with open("merged_notebook.ipynb", "w", encoding="utf-8") as f:
    f.write(merged)
