# Cora Dataset (Citation Network for Machine Learning Papers)

This repository contains a selection of the [Cora dataset](http://www.research.whizbang.com/data), commonly used in machine learning and graph-based research tasks.

---

## 📄 Overview

The **Cora dataset** consists of scientific publications on Machine Learning, categorized into one of the following seven classes:

- `Case_Based`
- `Genetic_Algorithms`
- `Neural_Networks`
- `Probabilistic_Methods`
- `Reinforcement_Learning`
- `Rule_Learning`
- `Theory`

Each publication is described using a bag-of-words representation, and citation links form a directed graph among the documents.

---

## 📊 Dataset Statistics

- **Total papers**: 2,708  
- **Vocabulary size**: 1,433 unique words  
  - (After stemming and removing stopwords)
  - Words with document frequency < 10 were removed

- **Each paper**:
  - Belongs to **one class**
  - Cites or is cited by **at least one other paper**

---

## 📁 Files in This Directory

### `cora.content`

This file contains content and metadata for each paper.  
Each line is structured as:


- `<paper_id>`: Unique identifier of the paper
- `<word_i>`: Binary indicator (1 or 0) showing presence or absence of word *i* in the paper
- `<class_label>`: Class name of the paper

> Example:
> ```
> 31336 0 0 1 1 ... 0 Neural_Networks
> ```

---

### `cora.cites`

This file defines the citation graph.  
Each line represents a citation link in the following format:


The direction of the edge is:  


> Example:
> ```
> 11234 11078
> ```
> means paper `11078` cites paper `11234`.

---

## 🔗 Use Cases

- Node classification
- Graph neural networks (GNNs)
- Citation network analysis
- Text classification

---

## 🧠 Commonly Used In

- GCN (Graph Convolutional Networks)
- GAT (Graph Attention Networks)
- Semi-supervised learning on graphs
- t-SNE visualization of embeddings

---

## 📚 Reference

Sen, P., Namata, G., Bilgic, M., Getoor, L., Galligher, B., & Eliassi-Rad, T. (2008).  
**Collective Classification in Network Data**  
*AI Magazine, 29(3), 93–106.*

---

## 🔗 Original Source

- [Cora dataset download](http://www.research.whizbang.com/data)

---

