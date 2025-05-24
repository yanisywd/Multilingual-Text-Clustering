# Multilingual Text Clustering

## Objective

Implement and compare different clustering algorithms for multilingual document analysis (French and English). The system combines transformer embeddings with TF-IDF features to create enhanced vector representations and automatically selects the best clustering method.

## Implemented Techniques

### Feature Extraction

- **XLM-RoBERTa embeddings** for semantic representation
- **TF-IDF features** with language-specific stopwords
- **Language detection** using langdetect
- **Feature combination** with weighted scaling

### Clustering Algorithms

- **Spectral Clustering** with nearest neighbors affinity
- **Hierarchical Clustering** using Ward linkage
- **K-Means** for optimal cluster number detection

### Evaluation Metrics

- **ARI** (Adjusted Rand Index)
- **NMI** (Normalized Mutual Information)
- **Silhouette Score**

### Visualization

- **t-SNE** dimensionality reduction
- **Cluster analysis** with document mapping
- **New document classification** visualization

## Technologies

- **Transformers**: XLM-RoBERTa (PyTorch)
- **Clustering**: Scikit-learn
- **NLP**: NLTK, LangDetect
- **Visualization**: Matplotlib, t-SNE

## Usage

```python
python clustering.py
```

Configure corpus paths in the script:

```python
base_paths = [
    '/path/to/Corpus_Francais',
    '/path/to/Corpus_Anglais'
]
```

## Output

- Performance comparison between clustering methods
- Cluster visualizations saved as PNG files
- Detailed cluster analysis and document classification
