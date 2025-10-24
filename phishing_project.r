# ===============================================
# Phishing Website Classification - Full Comparison Script (Fixed)
# ===============================================

# 1️⃣ Install & load required packages
packages <- c("farff", "randomForest", "caret", "rpart", "rpart.plot", 
              "pROC", "e1071", "class", "ggplot2", "corrplot", "dplyr")
for(p in packages){
  if(!require(p, character.only = TRUE)) install.packages(p, repos = "https://cloud.r-project.org/")
  library(p, character.only = TRUE)
}

# 2️⃣ Load dataset
data_path <- "Z:/r-projecct/projecct1/Training Dataset.arff"
data <- readARFF(data_path)

# 3️⃣ Inspect dataset
cat("Dataset dimensions:", dim(data), "\n")
str(data)
summary(data)

# 4️⃣ Convert target to factor
data$Result <- as.factor(data$Result)

# 5️⃣ Class distribution (original dataset)
class_table <- table(data$Result)

# Save all plots to PDF
pdf("phishing_classification_plots.pdf", width=10, height=7)

pie(class_table, main="Original Class Distribution", col=c("green","red"), labels=c("Legitimate","Phishing"))

# 6️⃣ Convert all factors to numeric for correlation (skip Result)
numeric_data <- data %>%
  select(-Result) %>%
  mutate(across(everything(), ~ as.numeric(as.character(.))))

corr_matrix <- cor(numeric_data)
corrplot(corr_matrix, method="color", tl.cex=0.6, title="Feature Correlation Heatmap")

# 7️⃣ Split dataset (80%-20%)
set.seed(123)
train_index <- createDataPartition(data$Result, p=0.8, list=FALSE)
train_set <- data[train_index, ]
test_set  <- data[-train_index, ]

# 8️⃣ Prepare numeric target for some models
train_set$Result_num <- ifelse(train_set$Result=="1",1,0)
test_set$Result_num  <- ifelse(test_set$Result=="1",1,0)

# =========================================
# 9️⃣ Random Forest
set.seed(123)
rf_model <- randomForest(Result ~ ., data=train_set, ntree=100, importance=TRUE)
rf_pred <- predict(rf_model, test_set)
rf_prob <- as.numeric(rf_pred)-1
rf_cm <- confusionMatrix(rf_pred, test_set$Result)
cat("\nRandom Forest Accuracy:", rf_cm$overall['Accuracy'], "\n")
varImpPlot(rf_model, main="Random Forest Feature Importance")

# =========================================
# 10️⃣ Decision Tree
dt_model <- rpart(Result ~ ., data=train_set, method="class")
dt_pred <- predict(dt_model, test_set, type="class")
dt_prob <- as.numeric(dt_pred)-1
dt_cm <- confusionMatrix(dt_pred, test_set$Result)
cat("\nDecision Tree Accuracy:", dt_cm$overall['Accuracy'], "\n")
rpart.plot(dt_model, main="Decision Tree")

# =========================================
# 11️⃣ Logistic Regression
lr_model <- glm(Result_num ~ . -Result, data=train_set, family=binomial)
lr_probs <- predict(lr_model, test_set, type="response")
lr_pred <- ifelse(lr_probs>0.5,1,0)
lr_cm <- confusionMatrix(as.factor(lr_pred), as.factor(test_set$Result_num))
cat("\nLogistic Regression Accuracy:", lr_cm$overall['Accuracy'], "\n")

# =========================================
# 12️⃣ k-Nearest Neighbors (k=5)
train_knn <- train_set[, -which(names(train_set) %in% c("Result","Result_num"))]
test_knn  <- test_set[, -which(names(test_set) %in% c("Result","Result_num"))]
train_labels <- train_set$Result
test_labels  <- test_set$Result
set.seed(123)
knn_pred <- knn(train=train_knn, test=test_knn, cl=train_labels, k=5)
knn_prob <- as.numeric(knn_pred)-1
knn_cm <- confusionMatrix(knn_pred, test_labels)
cat("\nkNN Accuracy:", knn_cm$overall['Accuracy'], "\n")

# =========================================
# 13️⃣ Support Vector Machine
svm_model <- svm(Result ~ ., data=train_set, probability=TRUE)
svm_pred <- predict(svm_model, test_set, probability=TRUE)
svm_prob <- attr(svm_pred,"probabilities")[, "1"]
svm_pred_class <- ifelse(svm_prob>0.5,1,0)
svm_cm <- confusionMatrix(as.factor(svm_pred_class), as.factor(test_set$Result_num))
cat("\nSVM Accuracy:", svm_cm$overall['Accuracy'], "\n")

# =========================================
# 14️⃣ Confusion Matrices
cat("\n=== Confusion Matrices ===\n")
cat("\nRandom Forest:\n"); print(rf_cm$table)
cat("\nDecision Tree:\n"); print(dt_cm$table)
cat("\nLogistic Regression:\n"); print(lr_cm$table)
cat("\nkNN:\n"); print(knn_cm$table)
cat("\nSVM:\n"); print(svm_cm$table)

# 15️⃣ ROC Curves and AUC
roc_rf  <- roc(test_set$Result_num, rf_prob)
roc_dt  <- roc(test_set$Result_num, dt_prob)
roc_lr  <- roc(test_set$Result_num, lr_probs)
roc_knn <- roc(test_set$Result_num, knn_prob)
roc_svm <- roc(test_set$Result_num, svm_prob)

plot(roc_rf, col="blue", main="ROC Curves - All Models")
lines(roc_dt, col="green")
lines(roc_lr, col="red")
lines(roc_knn, col="purple")
lines(roc_svm, col="orange")
legend("bottomright", legend=c("Random Forest","Decision Tree","Logistic Regression","kNN","SVM"),
       col=c("blue","green","red","purple","orange"), lwd=2)

cat("\nAUC Scores:\n")
cat("Random Forest:", auc(roc_rf), "\n")
cat("Decision Tree:", auc(roc_dt), "\n")
cat("Logistic Regression:", auc(roc_lr), "\n")
cat("kNN:", auc(roc_knn), "\n")
cat("SVM:", auc(roc_svm), "\n")

# 16️⃣ Model Comparison Table
accuracy_values <- c(rf_cm$overall['Accuracy'], dt_cm$overall['Accuracy'],
                     lr_cm$overall['Accuracy'], knn_cm$overall['Accuracy'], svm_cm$overall['Accuracy'])
auc_values <- c(auc(roc_rf), auc(roc_dt), auc(roc_lr), auc(roc_knn), auc(roc_svm))

comparison_table <- data.frame(
  Model = c("Random Forest","Decision Tree","Logistic Regression","kNN","SVM"),
  Accuracy = round(accuracy_values,4),
  AUC = round(auc_values,4)
)

comparison_table <- comparison_table %>% arrange(desc(Accuracy))
cat("\n=== Model Comparison Table (Ranked by Accuracy) ===\n")
print(comparison_table)

# Bar plot of Accuracy
ggplot(comparison_table, aes(x=reorder(Model, Accuracy), y=Accuracy, fill=Model)) +
  geom_bar(stat="identity") +
  coord_flip() +
  ggtitle("Model Accuracy Comparison") +
  theme_minimal() +
  theme(legend.position="none")

# =========================================
# 17️⃣ Pie charts of predicted class distribution
par(mfrow=c(2,3))  # 2 rows, 3 columns
pie(class_table, main="Original Class Distribution", col=c("green","red"), labels=c("Legitimate","Phishing"))
pie(table(rf_pred), main="Random Forest Predictions", col=c("green","red"), labels=c("Legitimate","Phishing"))
pie(table(dt_pred), main="Decision Tree Predictions", col=c("green","red"), labels=c("Legitimate","Phishing"))
pie(table(as.factor(lr_pred)), main="Logistic Regression Predictions", col=c("green","red"), labels=c("Legitimate","Phishing"))
pie(table(knn_pred), main="kNN Predictions", col=c("green","red"), labels=c("Legitimate","Phishing"))
pie(table(as.factor(svm_pred_class)), main="SVM Predictions", col=c("green","red"), labels=c("Legitimate","Phishing"))
par(mfrow=c(1,1))  # reset layout

# Close PDF
dev.off()
