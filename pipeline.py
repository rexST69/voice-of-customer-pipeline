"""
Text Analysis Pipeline
======================
Reads data (CSV or Excel), cleans it, and performs:
EDA, Network Analysis, Sentiment Analysis, Topic Modeling, Clustering.

Features:
- Auto-detects .csv or .xlsx
- Estimated Time Remaining (ETA)
- Multi-core processing

Usage:
python pipeline.py
"""

import sys
import re
import time
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import nltk
import os

from collections import Counter
from nltk.corpus import stopwords, wordnet, opinion_lexicon
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage
from wordcloud import WordCloud
from tqdm import tqdm

# Enable tqdm for pandas operations
tqdm.pandas()
warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION, change them accordingly to your requirement especially the custom removals
# ==============================================================================
CONFIG = {
    # Detects 'ai_dataset.csv' (from scraper) or 'ai_dataset.xlsx'
    "DATA_FILE": "xyz.csv", 
    "STOPWORDS_FILE": "stopwords.evaluation.xlsx",
    "MIN_TEXT_LENGTH": 50,
    "NUM_TOPICS": 8,
    "TOP_N_FREQ": 20,
    "NETWORK_THRESHOLD": 0.4,
    "SPARSE_THRESHOLD": 0.03,
    "CUSTOM_REMOVALS": ["youâ", "weâ", "it", "don't", "doesn't", "cant", "isnt", "https", "www", "com"],
    "N_CORES": -1
}

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def setup_nltk():
    resources = ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'opinion_lexicon']
    for res in resources:
        try:
            nltk.download(res, quiet=True)
        except: pass

def get_wordnet_pos(word):
    if not word: return wordnet.NOUN
    if word.endswith('ing'): return wordnet.VERB
    if word.endswith('ed'): return wordnet.VERB
    return wordnet.NOUN

def fast_clean_text(text, lemmatizer, stop_set):
    if not isinstance(text, str): return []
    # Remove URLs and non-alpha chars
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text.lower())
    
    try:
        tokens = nltk.word_tokenize(text)
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
        tokens = nltk.word_tokenize(text)
        
    tokens = [lemmatizer.lemmatize(t, get_wordnet_pos(t)) 
              for t in tokens if t not in stop_set and len(t) > 2]
    return tokens

def track_step(step_name, pbar):
    pbar.set_description(f"Processing: {step_name}")
    return time.time()

def end_step(start_time, pbar):
    elapsed = time.time() - start_time
    pbar.write(f"  ✓ Finished in {elapsed:.2f}s") 
    pbar.update(1)

# ==============================================================================
# MODULES
# ==============================================================================

def run_eda(all_words):
    word_counts = Counter(all_words)
    if not word_counts: return None
    
    # Freq Plot
    freq_df = pd.DataFrame(word_counts.most_common(CONFIG["TOP_N_FREQ"]), columns=['word', 'count'])
    plt.figure(figsize=(10, 6))
    sns.barplot(x='count', y='word', data=freq_df, hue='word', palette='viridis', legend=False)
    plt.title(f'Top {CONFIG["TOP_N_FREQ"]} Most Frequent Words')
    plt.tight_layout()
    plt.savefig("1_eda_frequency.png")
    plt.close()

    # Word Cloud
    wc = WordCloud(width=800, height=400, background_color='white', max_words=60).generate_from_frequencies(word_counts)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud')
    plt.savefig("1_eda_wordcloud.png")
    plt.close()
    return word_counts

def run_network(df, word_counts):
    if len(word_counts) < 2: return
    top_words = [w for w, c in word_counts.most_common(50)]
    try:
        vectorizer = CountVectorizer(vocabulary=top_words, binary=True)
        X = vectorizer.fit_transform(df['clean_text'])
        Xc = (X.T * X)
        Xc.setdiag(0)
        names = vectorizer.get_feature_names_out()
        df_co = pd.DataFrame(data=Xc.toarray(), columns=names, index=names)
        max_val = df_co.max().max()
        df_norm = df_co / max_val if max_val > 0 else df_co
        
        links = df_norm.stack().reset_index()
        links.columns = ['source', 'target', 'weight']
        links_filtered = links[(links['weight'] > (CONFIG["NETWORK_THRESHOLD"] / 2)) & (links['source'] < links['target'])]
        
        G = nx.from_pandas_edgelist(links_filtered, 'source', 'target', 'weight')
        if G.number_of_nodes() > 0:
            plt.figure(figsize=(12, 12))
            pos = nx.spring_layout(G, k=0.5, seed=42)
            nx.draw_networkx_nodes(G, pos, node_size=1000, node_color='lightblue')
            nx.draw_networkx_edges(G, pos, width=[d['weight']*5 for u, v, d in G.edges(data=True)], alpha=0.5)
            nx.draw_networkx_labels(G, pos, font_size=10)
            plt.axis('off')
            plt.savefig("2_network_diagram.png")
            plt.close()
    except Exception: pass

def run_sentiment(all_words, word_counts):
    pos_set = set(opinion_lexicon.positive())
    neg_set = set(opinion_lexicon.negative())
    pos_words = [w for w in all_words if w in pos_set]
    neg_words = [w for w in all_words if w in neg_set]
    
    pos_counts = Counter(pos_words)
    neg_counts = Counter(neg_words)
    data = []
    for w, c in pos_counts.most_common(15): data.append({'word': w, 'n': c, 'sentiment': 'positive'})
    for w, c in neg_counts.most_common(15): data.append({'word': w, 'n': c, 'sentiment': 'negative'})
    
    if data:
        df_sent = pd.DataFrame(data)
        g = sns.FacetGrid(df_sent, col="sentiment", sharey=False, height=5)
        g.map_dataframe(sns.barplot, x="n", y="word", hue="word", palette="coolwarm", legend=False)
        plt.subplots_adjust(top=0.9)
        g.fig.suptitle('Top Positive/Negative Words')
        plt.savefig("3_sentiment_bing.png")
        plt.close()

    try:
        from afinn import Afinn
        afinn = Afinn()
        scores = {w: afinn.score(w) * c for w, c in word_counts.items() if c > 5}
        scores = {k: v for k, v in scores.items() if v != 0}
        if scores:
            df_afinn = pd.DataFrame(list(scores.items()), columns=['word', 'contribution'])
            df_afinn['type'] = np.where(df_afinn['contribution'] > 0, 'Positive', 'Negative')
            df_afinn = df_afinn.sort_values(by='contribution', key=abs, ascending=False).head(20)
            plt.figure(figsize=(10, 6))
            sns.barplot(x='contribution', y='word', data=df_afinn, hue='type', dodge=False)
            plt.title('AFINN Sentiment Contribution')
            plt.tight_layout()
            plt.savefig("3_sentiment_afinn.png")
            plt.close()
    except ImportError: pass

def run_lda(df):
    try:
        tf_vectorizer = CountVectorizer(max_df=0.90, min_df=5, stop_words='english')
        tf = tf_vectorizer.fit_transform(df['clean_text'])

        print("\n  [INFO] Running LDA (See console for logs)...")
        lda = LatentDirichletAllocation(n_components=CONFIG["NUM_TOPICS"], random_state=1234, n_jobs=CONFIG["N_CORES"], verbose=1)
        lda.fit(tf)

        probs = lda.transform(tf)
        cols = [f"Topic {i+1}" for i in range(CONFIG["NUM_TOPICS"])]
        prob_df = pd.DataFrame(probs, columns=cols)
        prob_df.insert(0, 'id', df['id'].values)
        prob_df.to_csv("4_topic_probabilities.csv", index=False)
    except ValueError: pass

def run_clustering(df):
    try:
        tfidf = TfidfVectorizer(max_df=0.90, min_df=CONFIG["SPARSE_THRESHOLD"])
        tfidf_matrix = tfidf.fit_transform(df['clean_text'])
        dist = 1 - cosine_similarity(tfidf_matrix)
        linkage_matrix = linkage(dist, method='ward')

        plt.figure(figsize=(12, 6))
        dendrogram(linkage_matrix, labels=df['id'].values, leaf_rotation=90, leaf_font_size=8)
        plt.title('Hierarchical Clustering')
        plt.tight_layout()
        plt.savefig("5_clustering_dendrogram.png")
        plt.close()
    except Exception: pass

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=== Text Analysis Pipeline ===\n")
    
    # Auto-detect file
    if not os.path.exists(CONFIG["DATA_FILE"]):
        # Try alternate extensions
        base = CONFIG["DATA_FILE"].split('.')[0]
        if os.path.exists(f"{base}.xlsx"):
            CONFIG["DATA_FILE"] = f"{base}.xlsx"
        elif os.path.exists(f"{base}.csv"):
            CONFIG["DATA_FILE"] = f"{base}.csv"
        else:
            print(f"[Error] Could not find {CONFIG['DATA_FILE']} (or .csv/.xlsx variant).")
            sys.exit(1)

    pbar = tqdm(total=6, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} Steps", colour='green')
    
    t0 = track_step("Loading NLTK & Data", pbar)
    setup_nltk()
    try:
        if CONFIG['DATA_FILE'].endswith('.csv'):
            df = pd.read_csv(CONFIG['DATA_FILE'])
        else:
            df = pd.read_excel(CONFIG['DATA_FILE'])
            
        if 'Text' not in df.columns: raise ValueError("Missing 'Text' column")
    except Exception as e:
        print(f"\n[Error] {e}")
        sys.exit(1)
    end_step(t0, pbar)

    t0 = track_step("Cleaning Data", pbar)
    df = df[df['Text'].astype(str).str.len() > CONFIG["MIN_TEXT_LENGTH"]].copy()
    df['id'] = range(1, len(df) + 1)
    
    stop_set = set(stopwords.words('english'))
    try:
        custom = pd.read_excel(CONFIG["STOPWORDS_FILE"])
        stop_set.update(custom.iloc[:, 0].astype(str).str.lower().tolist())
    except: pass
    stop_set.update(CONFIG["CUSTOM_REMOVALS"])

    lemmatizer = WordNetLemmatizer()
    df['tokens'] = df['Text'].progress_apply(lambda x: fast_clean_text(x, lemmatizer, stop_set))
    df = df[df['tokens'].map(len) > 0]
    df['clean_text'] = df['tokens'].map(' '.join)
    all_words = [w for tokens in df['tokens'] for w in tokens]
    end_step(t0, pbar)

    t0 = track_step("EDA", pbar)
    word_counts = run_eda(all_words)
    end_step(t0, pbar)

    t0 = track_step("Network Analysis", pbar)
    run_network(df, word_counts)
    end_step(t0, pbar)

    t0 = track_step("Sentiment Analysis", pbar)
    run_sentiment(all_words, word_counts)
    end_step(t0, pbar)

    t0 = track_step("LDA & Clustering", pbar)
    run_lda(df)
    run_clustering(df)
    end_step(t0, pbar)

    pbar.close()
    print("\n=== Analysis Complete. Check folder for images. ===")

if __name__ == "__main__":
    main()
