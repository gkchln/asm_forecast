### Generalized Linear Model with Bernoulli response (Logistic Regression) ###

# Packages installation -----------------------------------------------------------------------
install.packages(c("car", "boot", "PRROC", "progress"))


# Imports -------------------------------------------------------------------------------------
library(car)
library(boot)
library(PRROC)
library(progress)

# Parameters ----------------------------------------------------------------------------------
scope <- 'BID'
input_path = sprintf('../../data/2_processed/ML_dataset_%s.csv', scope)

train_years <- c(2021)
test_years = c(2022)

target <- 'Result'

features_to_encode = c(
  'hour',
  'MargTech',
  'WorkDay',
  'Prov',
  'Tech'
)

other_features = c(
  'SC_PC1',
  'SC_PC2',
  'IN_PC1',
  'IN_PC2',
  'CT_PC1',
  'CT_PC2',
  'PurchMGP',
  'SellMGP',
  'SolarAngle',
  'DeclAngle',
  'PVnew',
  'PriceDiff'
)

features = c(other_features, features_to_encode)


# Functions -----------------------------------------------------------------------------------
APS <- function(model, df) {
  # Predict probabilities
  predicted_prob <- predict(model, newdata = df, type = "response")
  # Area under the Precision-Recall Curve (AU-PR) or Average Precision Score (APS)
  pr <- pr.curve(predicted_prob, weights.class0=df$Result, curve = TRUE)
  return(pr)
}

predict_proba_monthly_recal <- function(df, start_month, end_month) {
  
  observation_month <- as.integer(substr(df$idx, 1, 6))
  months <- sort(unique(observation_month))
  test_months <- months[months >= start_month & months <= end_month]
  y_probs_list <- c()
  
  pb <- progress_bar$new(format = "[:bar] :percent ETA: :eta", total = length(test_months))
  
  # Some observations are removed in the process because of unseen categories
  idx_removed <- c()
  result_idx <- c()
  
  for (test_month in test_months) {
    # For every month M, we take the training period as M-12 to M-1
    idx <- match(test_month, months)
    first <- idx - 12
    last <- idx - 1
    train_months <- months[first:last]
    train_data <- df[observation_month %in% train_months,]
    test_data <- df[observation_month == test_month,]
    
    # Removing the provinces that are in test but in train
    prov_to_remove <- setdiff(unique(test_data$Prov), unique(train_data$Prov))
    if (length(prov_to_remove) > 0) {
      exclude <- test_data$Prov %in% prov_to_remove
      test_data <- test_data[!(exclude),]
      idx_removed <- c(idx_removed, test_data[exclude, "idx"])
      warning(sprintf("Dropped %s samples in the test set corresponding to unseen Provinces", sum(exclude)))
    }
    
    model <- glm(
      Result ~ SC_PC1 + SC_PC2 + IN_PC1 + IN_PC2 + CT_PC1 + 
        CT_PC2 + PurchMGP + SellMGP + SolarAngle + 
        DeclAngle + PVnew + PriceDiff +
        WorkDay + hour + Tech + Prov,
      family = binomial,
      data = train_data,
    )
    
    y_probs <- as.vector(predict(model, newdata = test_data, type = "response"))
    #print(length(y_probs))
    #print(dim(test_data)[1])
    
    # Assuming your model is a binary classifier
    score <- APS(model, test_data)$auc.integral
    cat(sprintf("\nAverage Precision Score over %s samples for month %s is: %s\n\n", length(y_probs), test_month, round(score, 3)))
    
    y_probs_list <- c(y_probs_list, y_probs)
    result_idx <- c(result_idx, test_data$idx)
    pb$tick()
  }
  
  #result_index <- df[(observation_month %in% test_months) & !(df$idx %in% idx_removed), "idx"]
  #print(length(result_idx))
  #print(length(y_probs_list))
  return(data.frame(row.names = result_idx, y_probs = y_probs_list))
}

# Read data ---------------------------------------------------------------

df <- read.csv2(input_path, sep = ",", )

df[,other_features] <- lapply(df[,other_features], as.numeric)
df[,features_to_encode] <- lapply(df[,features_to_encode], as.factor)
df$Result <- as.logical(df$Result)

df[,other_features] <- scale(df[,other_features])

#train_df <- df[df$year %in% train_years,]
#test_df <- df[df$year %in% test_years,]

# HOTFIX: remove 'Olio' category from test_df for 2021 since it is not in train
#test_df <- test_df[test_df$MargTech != 'Olio',]

# Data exploration --------------------------------------------------------

# Set up the grid layout
#par(mfrow = c(4, 4))  # Adjust the number of rows and columns as needed
## Create boxplots for each variable grouped by "Result"
#for (feature in other_features) {
#  boxplot(train_df[,feature] ~ train_df$Result, main = feature, xlab = "Result", ylab = feature)
#}





# Modeling ----------------------------------------------------------------

## Logistic Regression ----------------------------------------------------
### Model with all regressors -----------------------------------------------
#logreg <- glm(
#  Result ~ SC_PC1 + SC_PC2 + IN_PC1 + IN_PC2 + CT_PC1 + 
#           CT_PC2 + PurchMGP + SellMGP + SolarAngle + 
#           DeclAngle + PVnew + PriceDiff +
#           WorkDay + hour + Tech + Prov + MargTech,
#           # I removed MargTech because it can happen that one level is not present in the train
#           # but in the test
#  family = binomial,
#  data = train_df,
#)
#
#summary(logreg)
#pr <- APS(logreg, test_df)
#plot(pr)


# Monthly Recalibration -----------------------------------------------------------------------
start_month <- 201901
end_month <- 202212

sub_df <- df[df$hour %in% c('9', '10') & df$Prov %in% c('Vercelli', 'Torino', 'Genova', 'Reggio Emilia'),]

timer = system.time({
  probs_df <- predict_proba_monthly_recal(df, start_month, end_month)
})
cat("Total elapsed time:", timer["elapsed"], "seconds\n")

write.csv(
  probs_df,
  file = sprintf('../%s/model_predictions/GLM_predicted_probs_monthly_recal_rolling_12m.csv', scope),
  row.names = TRUE
)




