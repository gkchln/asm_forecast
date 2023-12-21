### Linear Mixed Model for ASM Bid acceptance prediction ###

library(lme4)
library(nlme)
library(car)
library(boot)

find_best_prob_threshold <- function(predicted_prob, test_result) {
  f1_score_list = c()
  for (k in 1:99) {
    threshold <- k/100
    predicted_class <- ifelse(predicted_prob > threshold, 1, 0)
    if (sum(predicted_class) > 0) {
      # Create confusion matrix
      confusion_matrix <- table(Actual = test_result, Predicted = predicted_class)
      # Calculate F1 score
      precision <- confusion_matrix[2, 2] / sum(confusion_matrix[, 2])
      recall <- confusion_matrix[2, 2] / sum(confusion_matrix[2, ])
      f1_score <- 2 * precision * recall / (precision + recall)
    } else {
      f1_score <- 0
    }
    f1_score_list <- c(f1_score_list, f1_score)
  }
  plot(f1_score_list, type = "l", col = "darkred", lwd = 2,
       xlab = "100*threshold", ylab = "F1-scores", main = "F1-scores")
  
  best_f1 <- max(f1_score_list)
  best_threshold <- which.max(f1_score_list) / 100
  
  return(best_threshold)
}

test_model <- function(model, df) {
  # Predict probabilities
  predicted_prob <- predict(model, newdata = df, type = "response")
  
  # Get best threshold
  threshold <- find_best_prob_threshold(predicted_prob, df$Result)
  
  # Convert probabilities to binary predictions (0 or 1)
  predicted_class <- ifelse(predicted_prob > threshold, 1, 0)
  
  # Create confusion matrix
  confusion_matrix <- table(Actual = df$Result, Predicted = predicted_class)
  
  # Calculate precision, recall, and F1 score
  precision <- confusion_matrix[2, 2] / sum(confusion_matrix[, 2])
  recall <- confusion_matrix[2, 2] / sum(confusion_matrix[2, ])
  f1_score <- 2 * precision * recall / (precision + recall)
  
  # Print precision, recall, and F1 score
  cat("Precision:", precision, "\n")
  cat("Recall:", recall, "\n")
  cat("F1 Score:", f1_score, "\n")
  
  # Return the confusion matrix
  return(confusion_matrix)
}

glm_CV <- function(formula, df) {
  
  month_list <- unique(substr(df$idx, 4, 5))
  accuracy <- c()
  precision <- c()
  recall <- c()
  f1_score <- c()
  threshold <- c()
  
  for (month in month_list) {
    
    train_df <- df[substr(df$idx, 4, 5) != month,]
    test_df <- df[substr(df$idx, 4, 5) == month,]
    
    cat("Fitting leaving month", month, "out...\n")
    model <- glm(formula, data = train_df, family=binomial)
    
    cat("Testing month", month, ":\n")
    # Predict probabilities
    predicted_prob <- predict(model, newdata = test_df, type = "response")
    # Get best threshold
    thresh <- find_best_prob_threshold(predicted_prob, test_df$Result)
    predicted_class <- ifelse(predicted_prob > thresh, 1, 0)
    # Create confusion matrix
    confusion_matrix <- table(Actual = test_df$Result, Predicted = predicted_class)
    
    # Calculate precision, recall, and F1 score
    accuracy_score <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
    precision_score <- confusion_matrix[2, 2] / sum(confusion_matrix[, 2])
    recall_score <- confusion_matrix[2, 2] / sum(confusion_matrix[2, ])
    f1 <- 2 * precision_score * recall_score / (precision_score + recall_score)
    
    # Print precision, recall, and F1 score
    cat("Accuracy:", accuracy_score, "\n")
    cat("Precision:", precision_score, "\n")
    cat("Recall:", recall_score, "\n")
    cat("F1 Score:", f1, "\n")
    cat("Best threshold:", thresh, "\n\n")
    
    accuracy <- c(accuracy, accuracy_score)
    precision <- c(precision, precision_score)
    recall <- c(recall, recall_score)
    f1_score <- c(f1_score, f1)
    threshold <- c(threshold, thresh)
    
  }
  
  return(data.frame(accuracy, precision, recall, f1_score,
                    threshold, row.names = month_list))
}

# Read data ---------------------------------------------------------------

df <- read.csv2("train_db_dw.csv", sep = ",")
hold_out_df <- 

num_cols_idx <- c(3:10, 12:14, 18, 19)
fact_cols_idx <- c(2, 11, 15, 16, 17, 22)
df[,num_cols_idx] <- lapply(df[,num_cols_idx], as.numeric)
df[,fact_cols_idx] <- lapply(df[,fact_cols_idx], as.factor)
df$Result <- as.integer(df$Result)

df[,num_cols_idx] <- scale(df[,num_cols_idx])

train_df <- df[substr(df$idx, 4, 5) != '11',]
test_df <- df[substr(df$idx, 4, 5) == '11',]

# Data exploration --------------------------------------------------------

# Set up the grid layout
par(mfrow = c(4, 4))  # Adjust the number of rows and columns as needed
# Create boxplots for each variable grouped by "Result"
for (idx in num_cols_idx) {
  boxplot(df[,idx] ~ df$Result, main = colnames(df)[idx], xlab = "Result", ylab = colnames(df)[idx])
}





# Modeling ----------------------------------------------------------------

## Logistic Regression ----------------------------------------------------
### Model without Provinces -----------------------------------------------
logreg <- glm(
  Result ~ SC_PC1 + SC_PC2 + IN_PC1 + IN_PC2 + CT_PC1 + 
           CT_PC2 + PurchMGP + SellMGP + GNprice + SolarAngle + 
           DeclAngle + PV + Price + WorkDay + Tech,
  family = binomial,
  data = train_df,
)

summary(logreg)
conf_matrix <- test_model(model = logreg, df = test_df)
conf_matrix

logreg_CV <- glm_CV(
  Result ~ SC_PC1 + SC_PC2 + IN_PC1 + IN_PC2 + CT_PC1 + CT_PC2 +
    PurchMGP + SellMGP + GNprice + SolarAngle + DeclAngle + PV + Price +
    WorkDay + Tech,
  df
)

colMeans(logreg_CV)
boxplot(logreg_CV)


# We remove DeclAngle and SC_PC1 as they are not significant
logreg <- glm(
  Result ~ SC_PC2 + IN_PC1 + IN_PC2 + CT_PC1 + 
    CT_PC2 + PurchMGP + SellMGP + GNprice + SolarAngle + 
    PV + Price + WorkDay + Tech,
  family = binomial,
  data = train_df,
)

summary(logreg)
conf_matrix <- test_model(model = logreg, df = test_df)
conf_matrix

### Model with Provinces -----------------------------------------------
# Now trying to add Prov but factor variable with many (~30) levels
logreg2 <- glm(
  Result ~ SC_PC2 + IN_PC1 + IN_PC2 + CT_PC1 + 
    CT_PC2 + PurchMGP + SellMGP + GNprice + SolarAngle + 
    PV + Price + WorkDay + Tech + Prov, # Prov added
  family = binomial,
  data = train_df,
)

summary(logreg2)
conf_matrix2 <- test_model(model = logreg2, df = test_df)
conf_matrix2

### Model without Provinces and with PUs -----------------------------------
logreg2 <- glm(
  Result ~ SC_PC2 + IN_PC1 + IN_PC2 + CT_PC1 + 
    CT_PC2 + PurchMGP + SellMGP + GNprice + SolarAngle + 
    PV + Price + WorkDay + Tech + PU, # PU added
  family = binomial,
  data = train_df,
)

summary(logreg2)
# Removing one level which is not in the train set
conf_matrix2 <- test_model(model = logreg2,
                           df = test_df[test_df$PU != "UP_GOGLIO_2",])
conf_matrix2


## Mixed-Effect Logistic Regression ------------------------------------------
mlogreg <- glmer(
  Result ~ SC_PC1 + SC_PC2 + IN_PC1 + IN_PC2 + CT_PC1 + 
    CT_PC2 + PurchMGP + SellMGP + GNprice + SolarAngle + 
    DeclAngle + PV + Price + WorkDay + Tech + (1|Prov),
  family = binomial,
  data = train_df,
)

summary(mlogreg)
conf_matrix <- test_model(model = mlogreg, df = test_df)
conf_matrix

summary(glmm)
#confint(glmm, oldNames=TRUE)

qqnorm(resid(glmm))
qqline(resid(glmm), col='red', lwd=2)

residualPlot(glmm)

barplot(height = tapply(df$Result, df$PU, mean), beside = TRUE,
        col = rainbow(length(unique(df$Group))), legend.text = unique(df$Group),
        main = "Acceptance by PU", xlab = "PU", ylab = "Mean Acceptance")

mean_values <- aggregate(Result ~ PU, data = df, FUN = mean)

