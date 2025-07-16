# 📘 GCN_Project: Graph Convolutional Networks on Citation Datasets

This repository explores the use of **Graph Convolutional Networks (GCNs)** for **node classification** tasks on citation networks such as **Cora** and **Citeseer**.

Each dataset is organized in its own folder with dedicated code, outputs, and documentation. The goal is to compare performance, understand GCN behavior, and visualize embeddings on real-world graphs.

---

## 📂 Repository Structure


---

## 📁 Dataset-specific Folders

### 🔷 [Cora Dataset](./Cora%20Dataset)

This folder contains everything related to the **Cora citation network**:
- Raw dataset files (`.cites`, `.content`)
- Preprocessing and graph construction scripts
- GCN model implementation using PyTorch
- Training logs and visualizations
- [README](./Cora%20Dataset/README.md) with detailed explanation

### 🔶 [Citeseer Dataset](./Citeseer%20Dataset)

This folder contains all work done on the **Citeseer dataset**:
- Raw data files
- Node classification using GCN
- Code structure similar to Cora for consistency
- [README](./Citeseer%20Dataset/README.md) for specific commands and results

---

## 📑 Report

📄 The full project report including:
- Dataset analysis  
- GCN architecture overview  
- Training procedure  
- Experimental results and visualizations  
- Conclusions and future work  

Can be found here: [`report/report.pdf`](./Report.pdf)

---

## ⚙️ Environment Setup

All dependencies are included in the `environment.yml` file.

### To create and activate the Conda environment:
```bash
conda env create -f environment.yml
conda activate gcn_env

---

### ✅ Final Touch Checklist

Make sure your repo contains:
- `cora/README.md`
- `citeseer/README.md`
- `report/report.pdf`
- `environment.yml` (optional but helpful)
- `.gitignore` (for Python projects, include `__pycache__/`, `.ipynb_checkpoints/`, etc.)

Let me know if you’d like me to help write the `README.md` for the `cora/` or `citeseer/` folders too!
