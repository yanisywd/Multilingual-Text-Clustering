import os
import re
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from langdetect import detect
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.decomposition import PCA, TruncatedSVD
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings("ignore")

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def read_corpus(base_paths):
    corpus = []
    labels = []
    label_names = []
    file_names = []
    
    label_index = 0
    
    for base_path in base_paths:
        for cluster_name in sorted(os.listdir(base_path)):
            cluster_path = os.path.join(base_path, cluster_name)
            
            if not os.path.isdir(cluster_path):
                continue
                
            label_names.append(cluster_name)
            
            for file_name in os.listdir(cluster_path):
                file_path = os.path.join(cluster_path, file_name)
                
                if not os.path.isfile(file_path) or not file_path.endswith('.txt'):
                    continue
                    
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    try:
                        text = file.read()
                        text = re.sub(r'\s+', ' ', text).strip()
                        corpus.append(text)
                        labels.append(label_index)
                        file_names.append(file_name)
                    except Exception as e:
                        print(f"Error reading {file_path}: {str(e)}")
            
            label_index += 1
    
    return corpus, labels, label_names, file_names

def detect_languages(corpus):
    languages = []
    for doc in corpus:
        try:
            lang = detect(doc[:1000])
            languages.append(lang)
        except:
            languages.append('unknown')
    return languages

def extract_tfidf_features(corpus, languages):
    fr_docs = [doc for i, doc in enumerate(corpus) if languages[i].lower() == 'fr']
    en_docs = [doc for i, doc in enumerate(corpus) if languages[i].lower() == 'en']
    fr_indices = [i for i, lang in enumerate(languages) if lang.lower() == 'fr']
    en_indices = [i for i, lang in enumerate(languages) if lang.lower() == 'en']
    
    fr_stop = list(stopwords.words('french'))
    en_stop = list(stopwords.words('english'))
    
    fr_vectorizer = TfidfVectorizer(max_features=100, stop_words=fr_stop, ngram_range=(1, 2))
    en_vectorizer = TfidfVectorizer(max_features=100, stop_words=en_stop, ngram_range=(1, 2))
    
    all_features = np.zeros((len(corpus), 200))
    
    if fr_docs:
        fr_features = fr_vectorizer.fit_transform(fr_docs).toarray()
        print(f"French TF-IDF features shape: {fr_features.shape}")
        for i, idx in enumerate(fr_indices):
            all_features[idx, :100] = fr_features[i]
    
    if en_docs:
        en_features = en_vectorizer.fit_transform(en_docs).toarray()
        print(f"English TF-IDF features shape: {en_features.shape}")
        for i, idx in enumerate(en_indices):
            all_features[idx, 100:] = en_features[i]
    
    return all_features, fr_vectorizer, en_vectorizer

def get_transformer_embeddings(corpus, batch_size=8):
    # XLM-RoBERTa model for multilingual document embeddings
    print("Chargement du modèle XLM-RoBERTa...")
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    model = AutoModel.from_pretrained("xlm-roberta-base")
    
    vectors = []
    
    print("Génération des embeddings...")
    for i in range(0, len(corpus), batch_size):
        batch_texts = corpus[i:i+batch_size]
        
        truncated_texts = [text[:512] for text in batch_texts]
        
        inputs = tokenizer(truncated_texts, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        for j in range(len(truncated_texts)):
            doc_len = (inputs.attention_mask[j] == 1).sum().item()
            doc_vector = outputs.last_hidden_state[j, :doc_len, :].mean(dim=0)
            vectors.append(doc_vector.cpu().numpy())
        
        print(f"Traité {min(i+batch_size, len(corpus))}/{len(corpus)} documents")
    
    return np.array(vectors), model, tokenizer

def combine_features(transformer_vectors, tfidf_features, languages):
    # Feature fusion: combines transformer embeddings, TF-IDF features and language information
    print("Combinaison des caractéristiques...")
    
    transformer_scaler = StandardScaler()
    transformer_scaled = transformer_scaler.fit_transform(transformer_vectors)
    
    lang_features = np.zeros((len(languages), 2))
    for i, lang in enumerate(languages):
        if lang.lower() == 'fr':
            lang_features[i, 0] = 1
        elif lang.lower() == 'en':
            lang_features[i, 1] = 1
    
    tfidf_scaler = StandardScaler()
    tfidf_scaled = tfidf_scaler.fit_transform(tfidf_features)
    
    transformer_weight = 1.0
    tfidf_weight = 0.8
    lang_weight = 1.0
    
    combined = np.hstack([
        transformer_scaled * transformer_weight,
        tfidf_scaled * tfidf_weight,
        lang_features * lang_weight
    ])
    
    print(f"Combined feature vector shape: {combined.shape}")
    
    return combined, transformer_scaler, tfidf_scaler

def find_optimal_clusters(vectors, max_k=15):
    print("Recherche du nombre optimal de clusters...")
    silhouette_scores = []
    k_values = range(2, min(max_k, len(vectors)//2))
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(vectors)
        score = silhouette_score(vectors, cluster_labels)
        silhouette_scores.append(score)
        print(f"  k={k}: Silhouette Score = {score:.4f}")
    
    best_k = k_values[np.argmax(silhouette_scores)]
    print(f"Nombre optimal de clusters: {best_k}")
    
    return best_k

def perform_spectral_clustering(vectors, n_clusters):
    # Spectral clustering for complex, non-linear data relationships
    print("Exécution du clustering spectral...")
    
    spectral = SpectralClustering(
        n_clusters=n_clusters,
        assign_labels='discretize',
        random_state=42,
        affinity='nearest_neighbors',
        n_neighbors=min(5, len(vectors)-1)
    )
    
    cluster_labels = spectral.fit_predict(vectors)
    return cluster_labels

def perform_hierarchical_clustering(vectors, n_clusters):
    print("Exécution du clustering hiérarchique...")
    
    Z = linkage(vectors, method='ward')
    
    cluster_labels = fcluster(Z, n_clusters, criterion='maxclust')
    
    return cluster_labels - 1

def evaluate_clustering(vectors, true_labels, cluster_labels):
    ari = adjusted_rand_score(true_labels, cluster_labels)
    nmi = normalized_mutual_info_score(true_labels, cluster_labels)
    
    try:
        sil = silhouette_score(np.array(vectors), cluster_labels)
        print(f"Adjusted Rand Index: {ari:.4f}")
        print(f"Normalized Mutual Information: {nmi:.4f}")
        print(f"Silhouette Score: {sil:.4f}")
        return ari, nmi, sil
    except:
        print(f"Adjusted Rand Index: {ari:.4f}")
        print(f"Normalized Mutual Information: {nmi:.4f}")
        print(f"Silhouette Score: N/A")
        return ari, nmi, None

def visualize_clusters(vectors, cluster_labels, true_labels, label_names, file_names, languages):
    # Visualization of clustering results with t-SNE
    vectors = np.array(vectors)
    
    true_labels = np.array(true_labels)
    
    if vectors.shape[1] > 50:
        print("Réduction de dimensionnalité pour la visualisation...")
        n_components = min(30, len(vectors) - 1) 
        print(f"Utilisation de {n_components} composantes PCA")
        
        svd = TruncatedSVD(n_components=n_components)
        vectors_reduced = svd.fit_transform(vectors)
        explained_var = sum(svd.explained_variance_ratio_) * 100
        print(f"Variance expliquée: {explained_var:.2f}%")
    else:
        vectors_reduced = vectors
    
    perplexity = min(30, len(vectors) // 5)
    perplexity = max(5, perplexity)
    
    print(f"Visualisation t-SNE (perplexité={perplexity})...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    vectors_2d = tsne.fit_transform(vectors_reduced)
    
    plt.figure(figsize=(20, 16))
    
    plt.subplot(2, 2, 1)
    scatter = plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c=cluster_labels, cmap='viridis', s=80, alpha=0.7)
    plt.colorbar(scatter, label='Cluster')
    plt.title('Clusters Découverts', fontsize=14)
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    
    plt.subplot(2, 2, 2)
    class_colors = plt.cm.tab10(np.linspace(0, 1, len(label_names)))
    for i, label in enumerate(np.unique(true_labels)):
        if i < len(label_names):
            mask = true_labels == label
            plt.scatter(vectors_2d[mask, 0], vectors_2d[mask, 1], 
                       label=label_names[label], s=80, alpha=0.7, color=class_colors[i])
    
    plt.title('Vraies Classes (Répertoires)', fontsize=14)
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.legend(fontsize=10)
    
    plt.subplot(2, 2, 3)
    lang_colors = {'fr': 'blue', 'en': 'red', 'unknown': 'gray'}
    
    for lang in lang_colors:
        mask = np.array([l.lower() for l in languages]) == lang
        if np.any(mask):
            plt.scatter(vectors_2d[mask, 0], vectors_2d[mask, 1], 
                       label=lang.upper(), color=lang_colors[lang], s=80, alpha=0.7)
    
    plt.title('Séparation par Langue', fontsize=14)
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.legend(fontsize=10)
    
    plt.subplot(2, 2, 4)
    scatter = plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c=cluster_labels, cmap='viridis', s=100, alpha=0.7)
    
    for i, (x, y) in enumerate(vectors_2d):
        short_name = file_names[i].split('.')[0][:10]
        plt.annotate(short_name, (x, y), fontsize=8, alpha=0.8)
    
    plt.title('Clusters avec Noms de Documents', fontsize=14)
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    
    plt.tight_layout()
    plt.savefig('clusters_visualization_enhanced.png', dpi=300)
    
    try:
        plt.show()
    except:
        print("Affichage interactif non disponible. L'image a été enregistrée sous 'clusters_visualization_enhanced.png'")
    
    print("\nRépartition des documents dans les clusters:")
    cluster_map = {}
    for i, cluster in enumerate(cluster_labels):
        if cluster not in cluster_map:
            cluster_map[cluster] = []
        if true_labels[i] < len(label_names):
            cluster_map[cluster].append((file_names[i], label_names[true_labels[i]], languages[i]))
    
    for cluster_id, docs in sorted(cluster_map.items()):
        print(f"\nCluster {cluster_id}:")
        for doc_name, true_label, lang in docs:
            print(f"  - {doc_name} (Vrai: {true_label}, Langue: {lang.upper()})")
    
    print("\nDistribution des vraies classes dans chaque cluster:")
    for cluster_id in sorted(cluster_map.keys()):
        docs = cluster_map[cluster_id]
        total = len(docs)
        class_counts = {}
        lang_counts = {}
        
        for _, true_label, lang in docs:
            if true_label not in class_counts:
                class_counts[true_label] = 0
            class_counts[true_label] += 1
            
            if lang.lower() not in lang_counts:
                lang_counts[lang.lower()] = 0
            lang_counts[lang.lower()] += 1
        
        print(f"\nCluster {cluster_id} (Total: {total} documents):")
        for class_name, count in class_counts.items():
            percentage = (count / total) * 100
            print(f"  - {class_name}: {count} documents ({percentage:.1f}%)")
        
        print("  Répartition par langue:")
        for lang, count in lang_counts.items():
            percentage = (count / total) * 100
            print(f"    - {lang.upper()}: {count} documents ({percentage:.1f}%)")

def analyze_new_document(file_path, enhanced_vectors, best_labels, 
                        transformer_model, tokenizer, 
                        fr_vectorizer, en_vectorizer, 
                        transformer_scaler, tfidf_scaler,
                        labels, label_names, file_names, languages):
    # Classification of new documents using the trained model
    print(f"\nAnalyse du nouveau fichier: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        new_text = file.read()
        new_text = re.sub(r'\s+', ' ', new_text).strip()
    
    try:
        new_lang = detect(new_text[:1000])
    except:
        new_lang = 'unknown'
    print(f"Langue détectée: {new_lang.upper()}")
    
    truncated_text = new_text[:512]
    inputs = tokenizer(truncated_text, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = transformer_model(**inputs)
    
    doc_len = (inputs.attention_mask[0] == 1).sum().item()
    transformer_vector = outputs.last_hidden_state[0, :doc_len, :].mean(dim=0).cpu().numpy().reshape(1, -1)
    
    tfidf_features_new = np.zeros((1, 200))
    if new_lang.lower() == 'fr' and fr_vectorizer is not None:
        fr_features = fr_vectorizer.transform([new_text]).toarray()
        tfidf_features_new[0, :100] = fr_features[0]
    elif new_lang.lower() == 'en' and en_vectorizer is not None:
        en_features = en_vectorizer.transform([new_text]).toarray()
        tfidf_features_new[0, 100:] = en_features[0]
    
    lang_features_new = np.zeros((1, 2))
    if new_lang.lower() == 'fr':
        lang_features_new[0, 0] = 1
    elif new_lang.lower() == 'en':
        lang_features_new[0, 1] = 1
    
    transformer_scaled = transformer_scaler.transform(transformer_vector)
    tfidf_scaled = tfidf_scaler.transform(tfidf_features_new)
    
    transformer_weight = 1.0
    tfidf_weight = 0.8
    lang_weight = 1.0
    
    combined_new = np.hstack([
        transformer_scaled * transformer_weight,
        tfidf_scaled * tfidf_weight,
        lang_features_new * lang_weight
    ])
    
    distances = cdist(combined_new, enhanced_vectors)
    nearest_idx = np.argmin(distances[0])
    predicted_cluster = best_labels[nearest_idx]
    
    print(f"Le nouveau texte appartient au cluster: {predicted_cluster}")
    
    cluster_members = [(file_names[i], label_names[labels[i]], languages[i]) 
                      for i, label in enumerate(best_labels) 
                      if label == predicted_cluster]
    
    print("\nDétails du cluster:")
    for name, true_label, lang in cluster_members:
        print(f"  - {name} (Vrai: {true_label}, Langue: {lang.upper()})")
    
    total = len(cluster_members)
    class_counts = {}
    lang_counts = {}
    
    for _, true_label, lang in cluster_members:
        if true_label not in class_counts:
            class_counts[true_label] = 0
        class_counts[true_label] += 1
        
        if lang.lower() not in lang_counts:
            lang_counts[lang.lower()] = 0
        lang_counts[lang.lower()] += 1
    
    print(f"\nDistribution des classes dans le cluster {predicted_cluster} (Total: {total} documents):")
    for class_name, count in class_counts.items():
        percentage = (count / total) * 100
        print(f"  - {class_name}: {count} documents ({percentage:.1f}%)")
    
    print("  Répartition par langue:")
    for lang, count in lang_counts.items():
        percentage = (count / total) * 100
        print(f"    - {lang.upper()}: {count} documents ({percentage:.1f}%)")
    
    visualize_with_new_document(enhanced_vectors, combined_new, best_labels, 
                              predicted_cluster, labels, label_names, 
                              file_names, languages, new_lang)
    
    return predicted_cluster, combined_new

def visualize_with_new_document(vectors, new_vector, cluster_labels, 
                               predicted_cluster, true_labels, 
                               label_names, file_names, languages, new_lang):
    all_vectors = np.vstack([vectors, new_vector])
    
    if all_vectors.shape[1] > 50:
        n_components = min(30, all_vectors.shape[0] - 1)
        print(f"Utilisation de {n_components} composantes PCA pour la visualisation")
        
        svd = TruncatedSVD(n_components=n_components)
        vectors_reduced = svd.fit_transform(all_vectors)
        explained_var = sum(svd.explained_variance_ratio_) * 100
        print(f"Variance expliquée: {explained_var:.2f}%")
    else:
        vectors_reduced = all_vectors
    
    perplexity = min(30, all_vectors.shape[0] // 5)
    perplexity = max(5, perplexity)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    vectors_2d = tsne.fit_transform(vectors_reduced)
    
    existing_coords = vectors_2d[:-1]
    new_coords = vectors_2d[-1:]
    
    plt.figure(figsize=(15, 12))
    
    scatter = plt.scatter(existing_coords[:, 0], existing_coords[:, 1], 
                         c=cluster_labels, cmap='viridis', s=80, alpha=0.7)
    
    plt.scatter(new_coords[:, 0], new_coords[:, 1], marker='*', color='red', 
               s=300, label='Nouveau document')
    
    plt.annotate('Nouveau document', new_coords[0], 
                 xytext=(new_coords[0, 0] + 1, new_coords[0, 1] + 1),
                 fontsize=12, weight='bold', color='red')
    
    plt.colorbar(scatter, label='Cluster')
    plt.title('Classification du nouveau document', fontsize=16)
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('new_document_classification.png', dpi=300)
    
    try:
        plt.show()
    except:
        print("Affichage interactif non disponible. L'image a été enregistrée sous 'new_document_classification.png'")

def main():
    base_paths = [
        '/Users/yanis/Downloads/Corpus_Francais',
        '/Users/yanis/Downloads/Corpus_Anglais'
    ]
    
    print("Étape 1: Lecture des documents...")
    corpus, labels, label_names, file_names = read_corpus(base_paths)
    print(f"Nombre total de documents: {len(corpus)}")
    print(f"Nombre de classes: {len(set(labels))}")
    print(f"Classes: {label_names}")
    
    print("Détection des langues...")
    languages = detect_languages(corpus)
    
    print("\nÉtape 2: Création des représentations de documents avec transformers...")
    transformer_vectors, model, tokenizer = get_transformer_embeddings(corpus)
    
    print("\nÉtape 3: Extraction des caractéristiques TF-IDF...")
    tfidf_features, fr_vectorizer, en_vectorizer = extract_tfidf_features(corpus, languages)
    
    print("\nÉtape 4: Combinaison des caractéristiques...")
    enhanced_vectors, transformer_scaler, tfidf_scaler = combine_features(
        transformer_vectors, tfidf_features, languages)
    
    n_classes = len(set(labels))
    
    print(f"\nÉtape 5: Clustering spectral avec {n_classes} clusters...")
    spectral_labels = perform_spectral_clustering(enhanced_vectors, n_clusters=n_classes)
    
    print("\nÉtape 6: Évaluation du clustering spectral...")
    ari_s, nmi_s, sil_s = evaluate_clustering(enhanced_vectors, labels, spectral_labels)
    
    print(f"\nÉtape 7: Clustering hiérarchique avec {n_classes} clusters...")
    hierarchical_labels = perform_hierarchical_clustering(enhanced_vectors, n_clusters=n_classes)
    
    print("\nÉtape 8: Évaluation du clustering hiérarchique...")
    ari_h, nmi_h, sil_h = evaluate_clustering(enhanced_vectors, labels, hierarchical_labels)
    
    best_score = 0
    best_method = ""
    best_labels = None
    
    methods = {
        "Spectral": (ari_s, nmi_s, spectral_labels),
        "Hiérarchique": (ari_h, nmi_h, hierarchical_labels)
    }
    
    for method, (ari, nmi, labels_method) in methods.items():
        score = ari * 0.5 + nmi * 0.5
        if score > best_score:
            best_score = score
            best_method = method
            best_labels = labels_method
    
    print(f"\nLe clustering {best_method} a donné de meilleurs résultats.")
    
    print("\nÉtape 9: Visualisation des clusters...")
    visualize_clusters(enhanced_vectors, best_labels, labels, label_names, file_names, languages)
    
    print("\nÉtape 10: Prédiction pour un nouveau document...")
    new_file_path = '/Users/yanis/Desktop/boudra/text.txt'
    
    analyze_new_document(
        new_file_path, 
        enhanced_vectors, 
        best_labels,
        model, 
        tokenizer, 
        fr_vectorizer, 
        en_vectorizer,
        transformer_scaler, 
        tfidf_scaler,
        labels, 
        label_names, 
        file_names, 
        languages
    )
    
    print("\nProcessus terminé!")

if __name__ == "__main__":
    main()