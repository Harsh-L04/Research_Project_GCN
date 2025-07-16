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

