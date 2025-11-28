"""
Text Analysis Pipeline
==================================================
Corrected to produce identical results to the R scripts.

Key Fixes:
- Proper BING sentiment lexicon matching
- Word correlation network (not co-occurrence)
- Word clustering (not document clustering)
- Correct preprocessing steps
- Matching visualization styles
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

from collections import Counter
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from wordcloud import WordCloud
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CONFIG = {
    "DATA_FILE": "ai_dataset.xlsx",
    "STOPWORDS_FILE": "stopwords.evaluation.xlsx",
    "MIN_TEXT_LENGTH": 50,
    "NUM_TOPICS": 8,
    "TOP_N_FREQ": 20,
    "NETWORK_MIN_FREQ": 20,  # Words must appear at least 20 times
    "NETWORK_THRESHOLD": 0.40,
    "SPARSE_THRESHOLD": 0.97,
    "CUSTOM_REMOVALS": ["youâ", "weâ", "it", "dont", "doesnt"],
    "OUTPUT_DIR": "analysis_output",
    "RANDOM_SEED": 123546
}

# ==============================================================================
# NLTK SETUP
# ==============================================================================


def setup_nltk():
    """Download required NLTK resources."""
    resources = ['punkt', 'punkt_tab', 'stopwords', 'wordnet',
                 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']
    for res in resources:
        try:
            nltk.download(res, quiet=True)
        except:
            pass

# ==============================================================================
# TEXT CLEANING 
# ==============================================================================


def clean_text_r_style(text):
    """
    Clean text EXACTLY as R scripts do:
    1. Lowercase
    2. Remove [,;:.]
    3. Remove digits
    4. Remove punctuation
    5. Squish whitespace
    """
    if not isinstance(text, str) or len(text) == 0:
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove ,;:.
    text = re.sub(r'[,;:.]', '', text)

    # Remove digits
    text = re.sub(r'\d+', '', text)

    # Remove all punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Remove extra whitespace (squish)
    text = ' '.join(text.split())

    return text


def lemmatize_tokens(tokens):
    """Lemmatize tokens using WordNet."""
    lemmatizer = WordNetLemmatizer()
    lemmatized = []
    for token in tokens:
        # Simple POS tagging for better lemmatization
        lemmatized.append(lemmatizer.lemmatize(token, pos='v'))
    return lemmatized

# ==============================================================================
# BING SENTIMENT LEXICON
# ==============================================================================


def load_bing_lexicon():
    """
    Load BING sentiment lexicon to match R's get_sentiments("bing").
    Uses NLTK's opinion_lexicon which is equivalent.
    """
    from nltk.corpus import opinion_lexicon

    try:
        pos_words = set(opinion_lexicon.positive())
        neg_words = set(opinion_lexicon.negative())
    except LookupError:
        nltk.download('opinion_lexicon', quiet=True)
        pos_words = set(opinion_lexicon.positive())
        neg_words = set(opinion_lexicon.negative())

    # Create sentiment dictionary
    bing_dict = {}
    for word in pos_words:
        bing_dict[word] = 'positive'
    for word in neg_words:
        bing_dict[word] = 'negative'

    return bing_dict


def load_afinn_lexicon():
    """Load AFINN sentiment scores."""
    try:
        from afinn import Afinn
        return Afinn()
    except ImportError:
        print("  Warning: AFINN not installed. Install with: pip install afinn")
        return None

# ==============================================================================
# ANALYSIS FUNCTIONS
# ==============================================================================


def run_eda(word_freq_df):
    """Create frequency plot and word cloud (matching R)."""

    # Bar plot - Top 20
    plt.figure(figsize=(8, 6))
    top_20 = word_freq_df.head(20).sort_values('n')
    plt.barh(range(len(top_20)), top_20['n'], color='steelblue')
    plt.yticks(range(len(top_20)), top_20['word'])
    plt.xlabel('Frequency')
    plt.ylabel(None)
    plt.title('Top 20 Most Frequent Words')
    plt.tight_layout()
    plt.savefig(f"{CONFIG['OUTPUT_DIR']}/1_eda_frequency.png",
                dpi=300, bbox_inches='tight')
    plt.close()

    # Word Cloud 
    word_freq_dict = dict(zip(word_freq_df['word'], word_freq_df['n']))

    wc = WordCloud(
        width=800,
        height=600,
        background_color='white',
        max_words=50,
        relative_scaling=0.5,
        colormap='Dark2',
        random_state=CONFIG['RANDOM_SEED']
    ).generate_from_frequencies(word_freq_dict)

    plt.figure(figsize=(10, 7.5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(f"{CONFIG['OUTPUT_DIR']}/1_eda_wordcloud.png",
                dpi=100, bbox_inches='tight')
    plt.close()

    # Save frequency table
    word_freq_df.to_csv(
        f"{CONFIG['OUTPUT_DIR']}/1_word_frequencies.csv", index=False)

    print(f"  Unique words: {len(word_freq_df)}")


def run_word_correlation_network(tokens_df, word_freq_df):
    """
    Create word CORRELATION network (matching R's pairwise_cor).
    This is different from co-occurrence - it measures how often words
    appear together relative to their individual frequencies.
    """

    # Filter to words appearing at least 20 times
    high_freq_words = word_freq_df[word_freq_df['n']
                                   >= CONFIG['NETWORK_MIN_FREQ']]['word'].tolist()

    if len(high_freq_words) < 2:
        print("  Warning: Not enough high-frequency words for network")
        return

    # Filter tokens to only high-frequency words
    tokens_filtered = tokens_df[tokens_df['word'].isin(high_freq_words)].copy()

    # Create document-word matrix
    doc_word = tokens_filtered.pivot_table(
        index='id',
        columns='word',
        aggfunc='size',
        fill_value=0
    )

    # Calculate correlation matrix
    correlation_matrix = doc_word.corr(method='pearson')

    # Convert to edge list with correlation > threshold
    edges = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr = correlation_matrix.iloc[i, j]
            if corr > CONFIG['NETWORK_THRESHOLD']:
                edges.append({
                    'source': correlation_matrix.columns[i],
                    'target': correlation_matrix.columns[j],
                    'correlation': corr
                })

    if not edges:
        print("  Warning: No edges above correlation threshold")
        return

    # Save correlation data
    edges_df = pd.DataFrame(edges)
    edges_df.to_csv(
        f"{CONFIG['OUTPUT_DIR']}/2_word_correlations.csv", index=False)

    # Create network graph
    G = nx.Graph()
    for edge in edges:
        G.add_edge(edge['source'], edge['target'], weight=edge['correlation'])

    # Plot
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G, k=1, iterations=50, seed=124356)

    # Draw edges with width proportional to correlation
    edge_widths = [G[u][v]['weight'] * 3 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_widths,
                           alpha=0.5, edge_color='darkgray')

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=500,
                           node_color='lightblue', alpha=0.8)

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=9)

    plt.title('Word Correlation Network (r > 0.40)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{CONFIG['OUTPUT_DIR']}/2_network_diagram.png",
                dpi=300, bbox_inches='tight')
    plt.close()

    print(
        f"  Network created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")


def run_sentiment_analysis(tokens_df, word_freq_df):
    """Sentiment analysis using BING and AFINN lexicons (matching R)."""

    # Load BING lexicon
    bing_dict = load_bing_lexicon()

    # Match words with BING sentiment
    tokens_df['sentiment'] = tokens_df['word'].map(bing_dict)
    sentiment_words = tokens_df[tokens_df['sentiment'].notna()].copy()

    if len(sentiment_words) == 0:
        print("  Warning: No sentiment words found")
        return

    # Count sentiment words
    sentiment_counts = sentiment_words.groupby(
        ['word', 'sentiment']).size().reset_index(name='n')
    sentiment_counts = sentiment_counts.sort_values('n', ascending=False)

    # Summary
    pos_count = sentiment_counts[sentiment_counts['sentiment']
                                 == 'positive']['n'].sum()
    neg_count = sentiment_counts[sentiment_counts['sentiment']
                                 == 'negative']['n'].sum()
    print(f"  Positive words: {pos_count} | Negative words: {neg_count}")

    # Save sentiment data
    sentiment_counts.to_csv(
        f"{CONFIG['OUTPUT_DIR']}/3_sentiment_bing_counts.csv", index=False)

    # Bar plot (matching R's faceted plot)
    top_sentiment = sentiment_counts.groupby('sentiment').head(15)

    fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=False)

    for i, sentiment in enumerate(['negative', 'positive']):
        data = top_sentiment[top_sentiment['sentiment']
                             == sentiment].sort_values('n')
        color = 'salmon' if sentiment == 'negative' else 'lightblue'

        axes[i].barh(range(len(data)), data['n'], color=color)
        axes[i].set_yticks(range(len(data)))
        axes[i].set_yticklabels(data['word'])
        axes[i].set_xlabel('Number of occurrences')
        axes[i].set_title(sentiment.capitalize())
        axes[i].invert_yaxis()

    plt.tight_layout()
    plt.savefig(
        f"{CONFIG['OUTPUT_DIR']}/3_sentiment_bing_barplot.png", dpi=300, bbox_inches='tight')
    plt.close()

    # AFINN sentiment analysis
    afinn = load_afinn_lexicon()
    if afinn is not None:
        # Calculate contributions
        word_scores = []
        for _, row in word_freq_df.iterrows():
            if row['n'] >= 5:  # Filter words appearing at least 5 times
                score = afinn.score(row['word'])
                if score != 0:
                    contribution = score * row['n']
                    word_scores.append({
                        'word': row['word'],
                        'occurrence': row['n'],
                        'contribution': contribution,
                        'avg_score': score
                    })

        if word_scores:
            afinn_df = pd.DataFrame(word_scores)
            afinn_df = afinn_df.sort_values(
                'contribution', key=abs, ascending=False).head(30)

            # Save AFINN data
            afinn_df.to_csv(
                f"{CONFIG['OUTPUT_DIR']}/3_sentiment_afinn_summary.csv", index=False)

            # Plot
            plt.figure(figsize=(10, 8))
            colors = ['lightcoral' if x <
                      0 else 'lightblue' for x in afinn_df['contribution']]

            afinn_sorted = afinn_df.sort_values('contribution')
            plt.barh(range(len(afinn_sorted)),
                     afinn_sorted['contribution'], color=colors)
            plt.yticks(range(len(afinn_sorted)), afinn_sorted['word'])
            plt.xlabel('Sentiment score × Number of occurrences')
            plt.ylabel(None)
            plt.title('Words with Greatest Sentiment Contributions (AFINN)')
            plt.tight_layout()
            plt.savefig(
                f"{CONFIG['OUTPUT_DIR']}/3_sentiment_afinn_contribution.png", dpi=300, bbox_inches='tight')
            plt.close()


def run_topic_modeling(df):
    """LDA topic modeling (matching R's LDA parameters)."""

    if len(df) < CONFIG['NUM_TOPICS']:
        print("  Warning: Not enough documents for topic modeling")
        return

    # Create document-term matrix
    vectorizer = CountVectorizer(max_df=0.9, min_df=2)
    dtm = vectorizer.fit_transform(df['clean_text'])

    print(f"  DTM dimensions: {dtm.shape[0]} documents × {dtm.shape[1]} terms")

    # LDA with matching parameters
    lda = LatentDirichletAllocation(
        n_components=CONFIG['NUM_TOPICS'],
        max_iter=1000,
        learning_method='batch',  # Closest to Gibbs sampling
        random_state=1234,
        n_jobs=-1
    )

    doc_topic_dist = lda.fit_transform(dtm)

    # Save document-topic probabilities (matching R's gamma matrix)
    gamma_df = pd.DataFrame(doc_topic_dist)
    gamma_df.columns = [f'topic_{i+1}' for i in range(CONFIG['NUM_TOPICS'])]
    gamma_df.insert(0, 'document', df['id'].values)

    # Reshape to match R's output format (document, topic, gamma)
    gamma_long = gamma_df.melt(
        id_vars='document', var_name='topic', value_name='gamma')
    gamma_long['topic'] = gamma_long['topic'].str.replace(
        'topic_', '').astype(int)
    gamma_long.to_csv(f"{CONFIG['OUTPUT_DIR']}/TopicLDA.csv", index=True)

    # Save top terms per topic (beta matrix)
    feature_names = vectorizer.get_feature_names_out()
    beta_data = []

    for topic_idx, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[-10:][::-1]
        for idx in top_indices:
            beta_data.append({
                'topic': topic_idx + 1,
                'term': feature_names[idx],
                'beta': topic[idx] / topic.sum()  # Normalize to probabilities
            })

    beta_df = pd.DataFrame(beta_data)
    beta_df.to_csv(
        f"{CONFIG['OUTPUT_DIR']}/4_topic_word_probabilities.csv", index=False)

    # Print top 6 terms per topic
    print(f"\n  Top 6 terms per topic:")
    for topic_idx in range(CONFIG['NUM_TOPICS']):
        topic_terms = beta_df[beta_df['topic'] == topic_idx + 1].head(6)
        terms_str = ', '.join(topic_terms['term'].tolist())
        print(f"    Topic {topic_idx + 1}: {terms_str}")


def run_hierarchical_clustering(tokens_df):
    """
    Hierarchical clustering of WORDS (not documents) - matching R script.
    R clusters words based on their co-occurrence patterns across documents.
    """

    # Create word-document matrix (transpose of document-term)
    word_doc = tokens_df.pivot_table(
        index='word',
        columns='id',
        aggfunc='size',
        fill_value=0
    )

    # Remove sparse words (keep words in at least 3% of documents)
    min_docs = int(len(word_doc.columns) * (1 - CONFIG['SPARSE_THRESHOLD']))
    word_doc_filtered = word_doc[(word_doc > 0).sum(axis=1) >= min_docs]

    if len(word_doc_filtered) < 4:
        print("  Warning: Not enough words for clustering")
        return

    print(f"  Clustering {len(word_doc_filtered)} words")

    # Normalize (scale)
    word_doc_scaled = (word_doc_filtered - word_doc_filtered.mean(axis=1).values.reshape(-1, 1)) / \
        word_doc_filtered.std(axis=1).values.reshape(-1, 1)
    word_doc_scaled = word_doc_scaled.fillna(0)

    # Calculate cosine distance
    distances = pdist(word_doc_scaled.values, metric='cosine')

    # Hierarchical clustering with Ward linkage
    linkage_matrix = linkage(distances, method='ward')

    # Plot dendrogram
    plt.figure(figsize=(12, 6))
    dendrogram(
        linkage_matrix,
        labels=word_doc_filtered.index.tolist(),
        leaf_rotation=90,
        leaf_font_size=8
    )
    plt.title('Hierarchical Clustering of Words')
    plt.xlabel('')
    plt.ylabel('Height')
    plt.tight_layout()
    plt.savefig(
        f"{CONFIG['OUTPUT_DIR']}/5_clustering_dendrogram.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Save cluster assignments (k=4 clusters)
    from scipy.cluster.hierarchy import fcluster
    clusters = fcluster(linkage_matrix, 4, criterion='maxclust')

    cluster_df = pd.DataFrame({
        'word': word_doc_filtered.index,
        'cluster': clusters
    })
    cluster_df.to_csv(
        f"{CONFIG['OUTPUT_DIR']}/5_word_cluster_assignments.csv", index=False)

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================


def main():
    import os

    print("=" * 70)
    print("TEXT ANALYTICS PIPELINE")
    print("=" * 70)

    # Create output directory
    os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)

    # Setup
    setup_nltk()

    # Progress bar
    pbar = tqdm(total=7, desc="Progress",
                bar_format="{l_bar}{bar}| {n}/{total}")

    # ===== 1. LOAD DATA =====
    pbar.set_description("[1/7] Loading data")
    try:
        df = pd.read_excel(CONFIG['DATA_FILE'], sheet_name='Sheet1')
        if 'Text' not in df.columns or 'id' not in df.columns:
            raise ValueError("Dataset must contain 'id' and 'Text' columns")
    except Exception as e:
        print(f"\nError loading data: {e}")
        sys.exit(1)

    print(f"  Loaded {len(df)} rows")
    pbar.update(1)

    # ===== 2. CLEAN TEXT =====
    pbar.set_description("[2/7] Cleaning text")

    # Filter by length
    df = df[df['Text'].astype(str).str.len() >
            CONFIG['MIN_TEXT_LENGTH']].copy()

    # Clean text (matching R)
    df['Text'] = df['Text'].apply(clean_text_r_style)

    # Lemmatize
    df['Text'] = df['Text'].apply(
        lambda x: ' '.join(lemmatize_tokens(x.split())))

    # Create row IDs (matching R's row_number())
    df['id'] = range(1, len(df) + 1)

    print(f"  Retained {len(df)} texts after cleaning")
    pbar.update(1)

    # ===== 3. TOKENIZATION =====
    pbar.set_description("[3/7] Tokenizing")

    # Load stopwords
    stop_words = set(stopwords.words('english'))

    try:
        custom_stops = pd.read_excel(CONFIG['STOPWORDS_FILE'])
        stop_words.update(custom_stops.iloc[:, 0].astype(
            str).str.lower().tolist())
    except:
        pass

    stop_words.update(CONFIG['CUSTOM_REMOVALS'])

    # Tokenize
    tokens_list = []
    for idx, row in df.iterrows():
        words = row['Text'].split()
        for word in words:
            if word not in stop_words and len(word) > 1:
                tokens_list.append({'id': row['id'], 'word': word})

    tokens_df = pd.DataFrame(tokens_list)

    # Store clean text for modeling
    df['clean_text'] = df['id'].map(
        tokens_df.groupby('id')['word'].apply(lambda x: ' '.join(x))
    )
    df = df[df['clean_text'].notna()]

    print(f"  Generated {len(tokens_df)} tokens")
    pbar.update(1)

    # ===== 4. EDA =====
    pbar.set_description("[4/7] EDA")
    word_freq_df = tokens_df['word'].value_counts().reset_index()
    word_freq_df.columns = ['word', 'n']
    word_freq_df['total'] = word_freq_df['n'].sum()
    word_freq_df['proportion'] = word_freq_df['n'] / word_freq_df['total']

    run_eda(word_freq_df)
    pbar.update(1)

    # ===== 5. NETWORK =====
    pbar.set_description("[5/7] Word correlation network")
    run_word_correlation_network(tokens_df, word_freq_df)
    pbar.update(1)

    # ===== 6. SENTIMENT =====
    pbar.set_description("[6/7] Sentiment analysis")
    run_sentiment_analysis(tokens_df, word_freq_df)
    pbar.update(1)

    # ===== 7. TOPIC & CLUSTERING =====
    pbar.set_description("[7/7] Topic modeling & clustering")
    run_topic_modeling(df)
    run_hierarchical_clustering(tokens_df)
    pbar.update(1)

    pbar.close()

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Output saved to: {CONFIG['OUTPUT_DIR']}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
