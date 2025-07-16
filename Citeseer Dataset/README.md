# Citeseer Dataset (Citation Network for Scientific Publications)

This repository contains a processed version of the [Citeseer dataset](http://www.research.whizbang.com/data), widely used for graph-based machine learning research, especially in node classification tasks.

---

## 📄 Overview

The **Citeseer dataset** is a citation network of scientific publications categorized into one of six classes:

- `Agents`
- `AI`
- `DB`
- `IR`
- `ML`
- `HCI`

Each publication is represented as a bag-of-words vector, and citation relationships form directed edges in a citation graph.

---

## 📊 Dataset Statistics

- **Total papers**: 3,327  
- **Vocabulary size**: 3,703 unique words  
  - Words are preprocessed with stemming and stop-word removal  
  - Words with document frequency < 5 are removed

- **Each paper**:
  - Belongs to **one class**
  - Cites or is cited by **at least one other paper**

---

## 📁 Files in This Directory

### `citeseer.content`

This file contains the content features and class labels for each paper.  
Each line is formatted as:


- `<paper_id>`: Unique identifier for the paper
- `<word_i>`: Binary indicator (1 or 0) showing presence or absence of word *i* in the paper
- `<class_label>`: The label/class assigned to the paper

> Example:
> ```
> 1043711 0 1 0 ... 0 AI
> ```

---

### `citeseer.cites`

This file describes the citation graph. Each line represents a citation relationship in the format:


The direction of the edge is:  


> Example:
> ```
> 1043728 1043711
> ```
> means paper `1043711` cites paper `1043728`.

---

## 🔗 Use Cases

- Node classification
- Graph neural networks (GNNs)
- Citation graph modeling
- Semi-supervised learning

---

## 🧠 Commonly Used In

- GCN (Graph Convolutional Networks)
- GraphSAGE
- Label propagation
- Network embedding and visualization

---

## 📚 Reference

Sen, P., Namata, G., Bilgic, M., Getoor, L., Galligher, B., & Eliassi-Rad, T. (2008).  
**Collective Classification in Network Data**  
*AI Magazine, 29(3), 93–106.*

---

## 🔗 Original Source

- [Citeseer dataset download](http://www.research.whizbang.com/data)

---
