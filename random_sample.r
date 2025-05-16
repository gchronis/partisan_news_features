library(ggplot2)
library(tidyverse)
library(broom)
library(dplyr)

# control analysis

# What we want to do in this analysis is take all of the lemmas in our random sample list
# and calculate feature differences between left and right news sources.
# The null hypothesis is that there is no significant difference between feature values
# on the left and right. 

# To test this hypothesis we'll use the binder features predicted by Jwalanthi's 
# code, which uses a FFNN to project BERT embeddings into binder feature space. 
# We regress separately from the partisan label to each feature. 
# The goal is to predict from the feature value whether we are on the left and right. 

# We also run a Lasso regression model that includes all of the features as distinct variables
# and does the variable selection. 

# We have two different data structures: the tokens and the features. 
#   - the tokens contain the sentence and the partisan label (as 'cluster')
#   - the features contain one row for each features, so there are 65 rows per token.

# We have features for 1000+ instances of 300 random words. They are stored in a distinct file
# for each word. 


# For this analysis we don't care about the label of the lemma. We only care about the partisan label. 
# So we need our data in this shape:
#   - token_id | cluster (i.e. partisan label) | feature1 | feature 2 | feature 3


# for each word, we load it into memory and get it in the right shape, then we attach it to our big dataframe


tokens <- read_csv("/home/gsc685/partisan_features_3_23/data/sampled_sentences.csv")

lemmas <- unique(tokens$word)

# for each lemma, load the features if they exist and attach them to the features dataframe
# this is the big dataframe that will hold all of the features
features <- data.table::data.table()  # Initialize an empty data.table to store all features

for (lemma in lemmas) {
  feature_file <- paste0("/home/gsc685/partisan_features_3_23/jwalanthi_features/bert-base-uncased/", lemma, "_feature_vectors_roberta_buchanan_layer7.csv")
  print(feature_file )
  if (file.exists(feature_file)) {
    lemma_features <- data.table::fread(feature_file)  # Use fread for faster reading
    lemma_features[, word := lemma]  # Add the lemma as a column
    
    features <- rbind(features, lemma_features, fill = TRUE)  # Append to the big data.table
  }
}

# Print a glimpse of the combined features data.table
print(glimpse(features))
features <- features[, -1]  


d <- features %>%
  mutate(word = as.factor(word)) %>% # convert word to character
  mutate(cluster = as.factor(cluster)) # convert cluster to character



# Pivot the data from long to wide format
d.wide <- d %>%
  pivot_wider(names_from = feature, values_from = predicted_value)

# Print a glimpse of the transformed data
glimpse(d.wide)
 
######### run the regression

# Assume df is your dataset with 60 variables and a "Condition" column
# Convert condition to a factor if it's not already
#d.wide$word <- as.factor(d.wide$word)
#d.wide$cluster <- as.factor(d.wide$cluster)


print(unique(d.wide$cluster))  # See what levels exist for 'word' var
d.wide$word <- relevel(d.wide$cluster, ref = "cluster4")  # Set "cluster4" as the reference level if needed


d.extremes = d.wide %>%
  filter(cluster %in% c("0", "4")) %>% # filter out all besides hyper-left and hyper-right
  droplevels() # remove levels for the partisan labels we aren't analyzing



#########
# We want to look at partisan uses of the words. first we'll ignore the word and then we'll do a multiple regression to the word and the party
# and maybe the interaction? the interaction between word and party.
# 

# Run independent regressions for each variable and store results

# this time we'll look at partisan label

# iterate over all of the vars (columns) besides "word" and "sentence"
# higher coefficients mean that the value of that feature increases from hyper-left to hyper-right



results <- map_dfr(names(d.extremes)[!(names(d.extremes) %in% c("word", "token_id", "cluster"))], function(var) {
  model <- lm(reformulate("cluster", response = var), data = d.extremes)
  summary_model <- summary(model)
  print(summary_model)
  p_value <- coef(summary_model)["cluster4", "Pr(>|t|)"]  
  coeff <- coef(summary_model)["cluster4", "Estimate"]
  r_squared <- summary_model$r.squared
  adj_r_squared <- summary_model$adj.r.squared
  
  tibble(Variable = var, P_Value = p_value, coeff = coeff, r_squared = r_squared, adj_r_squared = adj_r_squared )
})

# Adjust p-values for multiple comparisons (optional, e.g., Bonferroni or FDR)
results <- results %>%
  mutate(P_Adjusted = p.adjust(P_Value, method = "bonferroni"))


# Print significant results (e.g., p < 0.05)
significant_results <- results %>% filter(P_Adjusted < 0.05)
print(significant_results)

write.csv(significant_results, "significant_features_regression_random.csv", row.names = FALSE)

#################################

# now we'll do a multiple regression
partisan_results <- map_dfr(names(d.extremes)[!(names(d.extremes) %in% c("word", "token_id", "cluster"))], function(var) {
  model <- lm(reformulate(response = var, termlabels = c("cluster", "word", "word:cluster")), data = d.extremes)
  summary_model <- summary(model)
  print(summary_model)
  
  # cluster4 (hyperpartisan right) variable
  right_coeff <- coef(summary_model)["cluster4", "Estimate"]  
  right_p_value <- coef(summary_model)["cluster4", "Pr(>|t|)"]


  # wordimmigrants (saying "undocumented immigrants") variable
  immigrants_coeff <- coef(summary_model)["wordimmigrants", "Estimate"]  
  immigrants_p_value <- coef(summary_model)["wordimmigrants", "Pr(>|t|)"]


  # interaction term: saying immigrants in right wing news
  inter_coeff <- coef(summary_model)["cluster4:wordimmigrants", "Estimate"]  
  inter_p_value <- coef(summary_model)["cluster4:wordimmigrants", "Pr(>|t|)"]
  
  r_squared <- summary_model$r.squared
  adj_r_squared <- summary_model$adj.r.squared  
  
  tibble(Variable = var, 
         right_coeff = right_coeff, 
         right_p_value = right_p_value, 
         immigrants_coeff = immigrants_coeff,
         immigrants_p_value = immigrants_p_value,
         inter_coeff = inter_coeff,
         inter_p_value = inter_p_value,
         r_squared = r_squared, adj_r_squared = adj_r_squared )
})


### What is the correlation between left and right partisanship and choice of alien or immigrant?
######

# first reshape the data to have one row per example sentence
sentences <- d.dedup %>%
  distinct(token_id, word, cluster)

# Convert categorical variable to numeric (0 & 1)
sentences$word_numeric <- as.numeric(as.factor(sentences$word)) - 1

# Compute point-biserial correlation
correlation_result <- cor.test(sentences$word_numeric, sentences$cluster)
# usage obviously highly correlated with partisanship

# visualize proportion of usages in a stacked bar plot
ggplot(sentences, aes(x = cluster, fill = word)) +
  geom_bar(position = "fill") +  # Stacked proportion bars
  labs(x = "Partisanship", y = "Proportion", fill = "word") +
  scale_y_continuous(labels = scales::percent) +  # Convert y-axis to percentage
  theme_minimal()


#########
## Polysemy analysis: If you use the word alien as left-wing, does its meaning differ significantly from when you say alien as right wing?

# first, count occurrences of each word-cluster combination
cooccurrence_counts <- sentences %>%
  count(word, cluster) %>%
  arrange(desc(n))  # sort by count

ggplot(cooccurrence_counts, aes(x = cluster, y = word, fill = n)) +
  geom_tile(color = "white") +  # White grid lines
  geom_text(aes(label = n), color = "black", size = 5) +  # Add count labels
  scale_fill_gradient(low = "white", high = "blue") +  # Color scale
  labs(title = "Co-occurrence of word choice and partisanship", x = "Cluster", y = "Word", fill = "Count") +
  theme_minimal()
# while usage of immigrants declines steadily from left to right, there is a distinct left usage of aliens that we dont see in center-left outlets (pretty small sample to look at though

##############################
# the question then, is what is the difference in meaning between left and right uses of the word alien?

# first get just the far left and right uses of the word aliens
d.extreme_aliens = d.wide %>%
  filter(cluster %in% c("0", "4")) %>% # filter out all besides hyper-left and hyper-right
  droplevels() # remove levels for the partisan labels we aren't analyzing

# Run independent regressions for each variable and store results
# higher coefficients mean that value increases from aliens to immigrants
polysemy_aliens_results <- map_dfr(names(d.extreme_aliens)[!(names(d.extreme_aliens) %in% c("word", "sentence", "cluster"))], function(var) {
  model <- lm(reformulate("cluster", response = var), data = d.extreme_aliens)
  summary_model <- summary(model)
  print(summary_model)
  p_value <- coef(summary_model)["cluster4", "Pr(>|t|)"]
  coeff <- coef(summary_model)["cluster4", "Estimate"]  
  r_squared <- summary_model$r.squared
  adj_r_squared <- summary_model$adj.r.squared
  
  tibble(Variable = var, P_Value = p_value, coeff = coeff, r_squared = r_squared, adj_r_squared = adj_r_squared )
})

# Adjust p-values for multiple comparisons (optional, e.g., Bonferroni or FDR)
polysemy_aliens_results <- polysemy_aliens_results %>%
  mutate(P_Adjusted = p.adjust(P_Value, method = "bonferroni"))
significant_polysemy_aliens_results <- polysemy_aliens_results %>% filter(P_Adjusted < 0.05)


# Adjust p-values for multiple comparisons (optional, e.g., Bonferroni or FDR)
polysemy_aliens_results <- polysemy_aliens_results %>%
  mutate(P_Adjusted = p.adjust(P_Value, method = "bonferroni"))
significant_polysemy_aliens_results <- polysemy_aliens_results %>% filter(P_Adjusted < 0.05)
# with adjustment for multiple comparisons, there is NO significant difference between left and right uses of immigrants

