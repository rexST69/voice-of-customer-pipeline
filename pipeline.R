# ==============================================================================
# FAST TEXT ANALYTICS PIPELINE (With ETA)
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

# CONFIGURATION changes these as per your use-case especially the custom removals
DATA_FILE <- "xyz.xlsx"
STOPWORDS_FILE <- "stopwords.evaluation.xlsx"
MIN_TEXT_LENGTH <- 50
NUM_TOPICS <- 8
CUSTOM_REMOVALS <- c("youâ", "weâ", "it", "don't", "doesn't")

# --- ETA Helper Function ---
estimate_eta <- function(start_time, current_step, total_steps) {
  elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
  avg_time <- elapsed / current_step
  remaining_steps <- total_steps - current_step
  eta_sec <- round(avg_time * remaining_steps, 1)
  message(paste("  > Estimated Time Remaining for entire pipeline:", eta_sec, "seconds"))
}

start_total <- Sys.time()
message("\n=== Starting Analysis ===")
pb <- txtProgressBar(min = 0, max = 6, style = 3)

# 1. LOAD
raw_text <- read_excel(DATA_FILE, sheet = 1)
setTxtProgressBar(pb, 1)
estimate_eta(start_total, 1, 6)

# 2. FAST CLEAN
clean_data <- raw_text %>%
  select(id, Text) %>%
  filter(str_length(Text) > MIN_TEXT_LENGTH) %>%
  mutate(Text = str_remove_all(str_to_lower(Text), "[^a-z\\s]"))

clean_data$Text <- lemmatize_strings(clean_data$Text, engine = 'hunspell')

data(stop_words)
custom_stops <- tryCatch(read_excel(STOPWORDS_FILE), error = function(e) NULL)
if (!is.null(custom_stops)) { all_stopwords <- bind_rows(stop_words, custom_stops) } else { all_stopwords <- stop_words }

text_tokens <- unnest_tokens(clean_data, input = Text, output = word) %>%
  filter(!word %in% CUSTOM_REMOVALS) %>%
  anti_join(all_stopwords, by = "word") %>%
  drop_na(word)

setTxtProgressBar(pb, 2)
estimate_eta(start_total, 2, 6)

# 3. EDA
word_freq <- text_tokens %>% count(word, sort = TRUE)

p1 <- word_freq %>% head(20) %>% mutate(word = reorder(word, n)) %>%
  ggplot(aes(x = n, y = word)) + geom_col(fill = "steelblue") + labs(title = "Top 20 Words")
ggsave("1_eda_frequency.png", plot = p1, width = 8, height = 6)

png("1_eda_wordcloud.png", width=800, height=600)
wordcloud(words = word_freq$word, freq = word_freq$n, min.freq = 2, max.words = 60, colors = brewer.pal(8, "Dark2"))
dev.off()
setTxtProgressBar(pb, 3)
estimate_eta(start_total, 3, 6)

# 4. NETWORK
top_words_net <- word_freq %>% head(40) %>% pull(word)
word_phi <- text_tokens %>% filter(word %in% top_words_net) %>% pairwise_cor(word, id, sort = TRUE)

p2 <- word_phi %>% filter(correlation > 0.40) %>% graph_from_data_frame() %>% 
  ggraph(layout = "fr") + geom_edge_link(aes(edge = correlation, width = correlation), edge_colour = "darkgray") + 
  geom_node_point(size = 4, color = "lightblue") + geom_node_text(aes(label = name), repel = TRUE) + theme_void()
ggsave("2_network_diagram.png", plot = p2, width = 10, height = 10)
setTxtProgressBar(pb, 4)
estimate_eta(start_total, 4, 6)

# 5. SENTIMENT
sentiment_counts <- text_tokens %>% inner_join(get_sentiments("bing"), by = "word") %>%
  count(word, sentiment, sort = TRUE) %>% group_by(sentiment) %>% slice_max(n, n = 15) %>% ungroup() %>% mutate(word = reorder(word, n))

p3 <- ggplot(sentiment_counts, aes(x = n, y = word, fill = sentiment)) +
  geom_col(show.legend = FALSE) + facet_wrap(~sentiment, scales = "free_y")
ggsave("3_sentiment_bing.png", plot = p3, width = 10, height = 6)
setTxtProgressBar(pb, 5)
estimate_eta(start_total, 5, 6)

# 6. LDA
dtm <- text_tokens %>% count(id, word) %>% cast_dtm(id, word, n)
lda_model <- LDA(x = dtm, k = NUM_TOPICS, method = "Gibbs", control = list(iter = 1000))
write.csv(tidy(lda_model, matrix = "gamma"), "4_topic_probabilities.csv", row.names = FALSE)

# Clustering
word_dtm_cluster <- text_tokens %>% group_by(id) %>% count(word) %>% cast_dtm(document = id, term = word, n)
dtm_sparse <- removeSparseTerms(word_dtm_cluster, sparse = 0.95) 

if (ncol(dtm_sparse) > 2) {
  dtm_scaled <- scale(dtm_sparse)
  dist_matrix <- proxy::dist(as.matrix(dtm_scaled), method = "cosine")
  hc_model <- hclust(dist_matrix, method = "ward.D2")
  png("5_clustering_dendrogram.png", width=1000, height=600)
  plot(hc_model, main = "Hierarchical Clustering")
  rect.hclust(hc_model, k = 4, border = "red")
  dev.off()
}
setTxtProgressBar(pb, 6)

close(pb)
message("\n=== Analysis Complete ===")
