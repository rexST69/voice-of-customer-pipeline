# ==============================================================================
# TEXT ANALYTICS PIPELINE
# ==============================================================================

library(tidyverse)
library(readxl)
library(stringr)
library(textstem)
library(hunspell)
library(tidytext)
library(wordcloud)
library(RColorBrewer)
library(igraph)
library(ggraph)
library(widyr)
library(topicmodels)
library(textdata)
library(reshape2)
library(tm)
library(proxy)
library(textclean)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
DATA_FILE <- "ai_dataset.xlsx"
STOPWORDS_FILE <- "stopwords.evaluation.xlsx"
MIN_TEXT_LENGTH <- 50
NUM_TOPICS <- 8
OUTPUT_DIR <- "analysis_output"

# Create output directory
if (!dir.exists(OUTPUT_DIR)) {
  dir.create(OUTPUT_DIR, recursive = TRUE)
}

save_output <- function(filename) {
  file.path(OUTPUT_DIR, filename)
}

# ==============================================================================
# START ANALYSIS
# ==============================================================================
start_total <- Sys.time()
message("\n=== Starting Text Analytics Pipeline ===")
pb <- txtProgressBar(min = 0, max = 7, style = 3)

# ==============================================================================
# 1. DATA LOADING
# ==============================================================================
message("\n[1/7] Loading data...")
Text <- read_excel(DATA_FILE, sheet = "Sheet1")
message(sprintf("  Loaded %d rows", nrow(Text)))

setTxtProgressBar(pb, 1)

# ==============================================================================
# 2. DATA CLEANING & PREPROCESSING 
# ==============================================================================
message("\n[2/7] Cleaning and preprocessing...")

# Filter by length
Text <- Text %>%
  select(id, Text) %>%
  filter(str_length(Text) > MIN_TEXT_LENGTH)

# Cleaning - EXACTLY as original scripts
Text$Text <- str_to_lower(Text$Text, locale = "en")
Text$Text <- str_replace_all(Text$Text, "[,;:.]", "")
Text$Text <- str_replace_all(Text$Text, "[:digit:]", "")
Text$Text <- str_replace_all(Text$Text, "[[:punct:]]", "")
Text$Text <- str_squish(Text$Text)

# Lemmatization 
Text$Text <- lemmatize_strings(Text$Text, engine = "hunspell")

# Token replacement (optional - customize as needed)
Text$Text <- replace_tokens(Text$Text,
  tokens = c("youâ", "weâ", "it", "dont", "doesnt"),
  replacement = c("", "", "", "", "")
)

message(sprintf("  Retained %d texts after cleaning", nrow(Text)))

setTxtProgressBar(pb, 2)

# ==============================================================================
# 3. TOKENIZATION 
# ==============================================================================
message("\n[3/7] Tokenizing and removing stopwords...")

# Create tidy format - CRITICAL: Use row_number() for id
tidydata <- tibble(Text)
tidydata <- tidydata %>%
  mutate(id = row_number())

# Load stopwords
data(stop_words)
stopwords_ev <- tryCatch(
  {
    read_excel(STOPWORDS_FILE)
  },
  error = function(e) {
    data.frame(word = character(0), lexicon = character(0))
  }
)

stopwords_all <- stop_words %>% bind_rows(stopwords_ev)

# Tokenize 
Text.tokens <- unnest_tokens(tidydata, input = Text, output = word, to_lower = TRUE) %>%
  anti_join(stopwords_all, by = "word") %>%
  drop_na(word)

message(sprintf("  Generated %d tokens", nrow(Text.tokens)))

setTxtProgressBar(pb, 3)

# ==============================================================================
# 4. EDA - WORD FREQUENCY & WORD CLOUD 
# ==============================================================================
message("\n[4/7] Creating EDA visualizations...")

# Word frequency with proportion
word.freq1 <- Text.tokens %>%
  count(word, sort = TRUE) %>%
  mutate(total = sum(n)) %>%
  mutate(proportion = n / total)

message(sprintf("  Unique words: %d", nrow(word.freq1)))

# Bar plot - Top 20 words 
p1 <- word.freq1 %>%
  top_n(20, n) %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(x = n, y = word)) +
  geom_col(fill = "steelblue") +
  labs(x = "Frequency", y = NULL, title = "Top 20 Most Frequent Words")

ggsave(save_output("1_eda_frequency.png"), plot = p1, width = 8, height = 6, dpi = 300)

# Word cloud
set.seed(123546)
png(save_output("1_eda_wordcloud.png"), width = 800, height = 600)
word.freq1 %>%
  with(wordcloud(word, n,
    max.words = 50,
    scale = c(5, 0.8),
    random.order = FALSE,
    rot.per = 0.00,
    colors = brewer.pal(8, "Dark2")
  ))
dev.off()

# Save frequency table
write_csv(word.freq1, save_output("1_word_frequencies.csv"))

setTxtProgressBar(pb, 4)

# ==============================================================================
# 5. WORD NETWORK ANALYSIS
# ==============================================================================
message("\n[5/7] Building word correlation network...")

# Pairwise correlation 
word.phi <- Text.tokens %>%
  group_by(word) %>%
  filter(n() >= 20) %>% # CRITICAL: Filter words appearing at least 20 times
  pairwise_cor(word, id, sort = TRUE)

# Network visualization 
set.seed(124356)
a <- arrow(angle = 30, length = unit(0.1, "inches"), ends = "last", type = "open")

p2 <- word.phi %>%
  filter(correlation > 0.40) %>%
  graph_from_data_frame() %>%
  ggraph(layout = "fr") +
  geom_edge_link(aes(edge_alpha = correlation, edge_width = correlation),
    edge_colour = "darkgray", arrow = a
  ) +
  geom_node_point(size = 5, color = "lightblue") +
  geom_node_text(aes(label = name),
    repel = TRUE,
    point.padding = unit(0.2, "lines")
  ) +
  theme_void() +
  labs(title = "Word Correlation Network (r > 0.40)")

ggsave(save_output("2_network_diagram.png"), plot = p2, width = 10, height = 10, dpi = 300)

# Save correlation data
write_csv(
  word.phi %>% filter(correlation > 0.40),
  save_output("2_word_correlations.csv")
)

setTxtProgressBar(pb, 5)

# ==============================================================================
# 6. SENTIMENT ANALYSIS 
# ==============================================================================
message("\n[6/7] Performing sentiment analysis...")

# BING Sentiment - Word cloud comparison
Text.match <- Text.tokens %>%
  inner_join(get_sentiments("bing"), by = "word") %>%
  count(word, sentiment, sort = TRUE) %>%
  acast(word ~ sentiment, value.var = "n", fill = 0)

# Comparison cloud 
png(save_output("3_sentiment_comparison_cloud.png"), width = 800, height = 600)
comparison.cloud(Text.match,
  max.words = 60,
  scale = c(5, 1.2),
  random.order = FALSE,
  rot.per = 0.35,
  colors = brewer.pal(8, "Dark2")
)
dev.off()

# Bar diagram
Text.match.count <- Text.tokens %>%
  inner_join(get_sentiments("bing"), by = "word") %>%
  count(word, sentiment, sort = TRUE) %>%
  ungroup()

p3 <- Text.match.count %>%
  group_by(sentiment) %>%
  slice_max(n, n = 15) %>%
  ungroup() %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(x = n, y = word, fill = sentiment)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~sentiment, scales = "free_y") +
  labs(
    x = "Number of occurrences", y = NULL,
    title = "Top 15 Sentiment Words"
  )

ggsave(save_output("3_sentiment_bing_barplot.png"), plot = p3, width = 10, height = 6, dpi = 300)

# AFINN Sentiment - Score analysis
Text.sentiment <- Text.tokens %>%
  inner_join(get_sentiments("afinn"), by = "word")

Text.sentiment.summary <- Text.sentiment %>%
  group_by(word) %>%
  summarise(
    occurrence = n(),
    contribution = sum(value),
    Avg_score = sum(value) / n()
  ) %>%
  ungroup() %>%
  filter(occurrence >= 5)

# Contribution plot 
p4 <- Text.sentiment.summary %>%
  slice_max(abs(contribution), n = 30) %>%
  mutate(word = reorder(word, contribution)) %>%
  ggplot(aes(contribution, word, fill = contribution > 0)) +
  geom_col(show.legend = FALSE) +
  labs(
    x = "Sentiment score * Number of occurrences",
    y = NULL,
    title = "Words with Greatest Sentiment Contributions"
  )

ggsave(save_output("3_sentiment_afinn_contribution.png"), plot = p4, width = 10, height = 8, dpi = 300)

# Save sentiment data
write_csv(Text.match.count, save_output("3_sentiment_bing_counts.csv"))
write_csv(Text.sentiment.summary, save_output("3_sentiment_afinn_summary.csv"))

message(sprintf(
  "  Positive words: %d | Negative words: %d",
  sum(Text.match.count$sentiment == "positive"),
  sum(Text.match.count$sentiment == "negative")
))

setTxtProgressBar(pb, 6)

# ==============================================================================
# 7. TOPIC MODELING & CLUSTERING 
# ==============================================================================
message("\n[7/7] Topic modeling and clustering...")

# DTM for LDA 
Text.dtm <- Text.tokens %>%
  count(id, word) %>%
  cast_dtm(id, word, n)

message(sprintf(
  "  DTM dimensions: %d documents x %d terms",
  nrow(Text.dtm), ncol(Text.dtm)
))

# LDA Topic Modeling 
if (nrow(Text.dtm) >= NUM_TOPICS && ncol(Text.dtm) >= NUM_TOPICS) {
  message(sprintf("  Running LDA with %d topics...", NUM_TOPICS))

  Text.lda <- LDA(
    x = Text.dtm,
    k = NUM_TOPICS,
    method = "Gibbs",
    control = list(alpha = 1, delta = 0.1, seed = 1234)
  )

  # Top terms per topic
  top_terms <- terms(Text.lda, 6)
  print(top_terms)

  # Extract beta (word probabilities)
  keyword.prob <- tidy(Text.lda, matrix = "beta") %>%
    group_by(topic) %>%
    arrange(desc(beta)) %>%
    top_n(6, beta) %>%
    ungroup() %>%
    arrange(topic, desc(beta))

  # Extract gamma (document-topic probabilities)
  topic.prob <- tidy(Text.lda, matrix = "gamma")

  # Save
  write.csv(as.data.frame(topic.prob), save_output("TopicLDA.csv"), row.names = TRUE)
  write_csv(keyword.prob, save_output("4_topic_word_probabilities.csv"))

  message("  LDA complete")
} else {
  message("  Warning: Insufficient data for topic modeling")
}

# Hierarchical Clustering 
message("  Performing hierarchical clustering...")

word.freq2 <- Text.tokens %>%
  group_by(id) %>%
  count(word, sort = TRUE)

word.dtm <- word.freq2 %>%
  cast_dtm(document = id, term = word, n)

# Remove sparse terms
word.sdtm <- removeSparseTerms(word.dtm, sparse = 0.97)

if (ncol(word.sdtm) > 2 && nrow(word.sdtm) > 2) {
  # Scale and distance 
  word.sdtm.s <- scale(word.sdtm, scale = TRUE)

  # Transpose for word clustering (t())
  d <- proxy::dist(as.matrix(t(word.sdtm.s)), method = "cosine")

  # Hierarchical clustering
  set.seed(124356)
  hc <- hclust(d, method = "ward.D2")

  # Plot dendrogram
  png(save_output("5_clustering_dendrogram.png"), width = 1000, height = 600)
  plot(hc, main = "Hierarchical Clustering of Words", xlab = "", sub = "")
  groups <- cutree(hc, k = 4)
  rect.hclust(hc, k = 4, border = "red")
  dev.off()

  # Save cluster assignments
  cluster_df <- data.frame(
    word = names(groups),
    cluster = groups
  )
  write_csv(cluster_df, save_output("5_word_cluster_assignments.csv"))

  message("  Clustering complete")
} else {
  message("  Warning: Insufficient features for clustering")
}

setTxtProgressBar(pb, 7)

# ==============================================================================
# ANALYSIS COMPLETE
# ==============================================================================
close(pb)
end_total <- Sys.time()
duration <- as.numeric(difftime(end_total, start_total, units = "secs"))

message("\n", paste(rep("=", 70), collapse = ""))
message("ANALYSIS COMPLETE")
message(paste(rep("=", 70), collapse = ""))
message(sprintf(
  "Total runtime: %.2f seconds (%.2f minutes)",
  duration, duration / 60
))
message(sprintf("Output saved to: %s/", OUTPUT_DIR))
message(paste(rep("=", 70), collapse = ""))

# Generate summary report
summary_stats <- data.frame(
  metric = c(
    "Total Documents", "Processed Documents", "Total Tokens",
    "Unique Words", "Number of Topics", "Runtime (seconds)"
  ),
  value = c(
    nrow(read_excel(DATA_FILE, sheet = "Sheet1")),
    nrow(tidydata),
    nrow(Text.tokens),
    nrow(word.freq1),
    NUM_TOPICS,
    round(duration, 2)
  )
)

write_csv(summary_stats, save_output("0_analysis_summary.csv"))
