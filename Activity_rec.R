library(data.table)
#Mltinomial logistic regression
library(nnet)
#Random forest
library(randomForest)
#Decision tree
library(rpart)
#Libraries for knn
library(class)
#SVM
library(e1071)
#XgbBoosting
library(xgboost)
library(Matrix)
library(magrittr)
#Neural Networks
library(keras)
library(tensorflow)

#------------------------Merging datasets--------------------------------

setwd("S2_Dataset/")
reading_data_1 <- function(){
  path <- "C:\\Users\\Lukas\\Desktop\\R2/Activity_recognition/S2_Dataset/"
  files <- list.files(path, "p")
  l <- lapply(files, fread, header = FALSE, sep = ",", stringsAsFactors = FALSE, showProgress = FALSE)
  data <- do.call(rbind, l)
  return(data)
}
data <- as.data.frame(reading_data_1())

#------------------------Preparing data----------------------------------

for(i in 1:8){
  if (i==5) {
    next
  }
  data[ ,i] <- as.numeric(data[ ,i])
}

names(data) <- c('Time','Acc.Front','Acc.vert','Acc.Lat','id','RSSI','Phase','Freq','Activity_Label')

data[,9] <- as.numeric(data[,9]) -1 #Sumazinamas vienas lygis del NN, XgbBoost
data[,5] <- as.numeric(data[,5]) -1 #Sumazinamas vienas lygis del NN, XgbBoost
data$Activity_Label <- as.factor(data$Activity_Label)
data$id <- as.factor(data$id)

#----------------------Data particion-----------------------------------

set.seed(1234)
ind <- sample(2, nrow(data), replace = T, prob = c(0.8, 0.2))
train <- data[ind == 1, ]
test <- data[ind==2, ]

#---------------------Multinomial logistic regression-------------------

mlg_model <- multinom(Activity_Label ~ .,data = train)
mlg_model

#Predincted probobility on train data
predict(mlg_model, train, type = "prob")

#Confusion matrix on trained data
mlg_cm_tr <- confusionMatrix(predict(mlg_model, train), train$Activity_Label)

#Confusion matrix on test data
mlg_cm_te <- confusionMatrix(predict(mlg_model,test), test$Activity_Label)

#Accuracy on train data
sum(diag(mlg_cm_tr$table)/sum(mlg_cm_tr$table))

#Accuracy on test data
sum(diag(mlg_cm_te$table)/sum(mlg_cm_te$table))


#two tail z - test(if the p value is small then the confidence level is high - so that variable is significant)
z_test <- summary(mlg_model)$coefficients/summary(mlg_model)$standard.errors
(1 - pnorm(abs(z_test), 0, 1)) * 2

#F1 score
mlg_cm_te$byClass[,7]
mlg_f1 <- mean(mlg_cm_te$byClass[,7])

#---------------------Random forest--------------------------------------

rf_model <- randomForest(Activity_Label~.,data = train)
rf_model

#Confusion matrix on trained data
rf_cm_tr <- confusionMatrix(predict(rf_model, train), train$Activity_Label)

#Confusion matrix on test data
rf_cm_te <- confusionMatrix(predict(rf_model,test), test$Activity_Label)

#Accuracy on train data
sum(diag(rf_cm_tr)/sum(rf_cm_tr))

#Accuracy on test data
sum(diag(rf_cm_te)/sum(rf_cm_te))

#F1 score
rf_cm_te$byClass[,7]
rf_f1 <- mean(rf_cm_te$byClass[,7])
#---------------------Decision tree--------------------------------------

dt_model <- rpart(Activity_Label ~.,data = train)

#Confusion matrix on trained data
dt_cm_tr <- confusionMatrix(predict(dt_model, train, type="class"), train$Activity_Label)

#Confusion matrix on test data
dt_cm_te <- confusionMatrix(predict(dt_model, test, type="class"), test$Activity_Label)

#Accuracy on train data
sum(diag(dt_cm_tr)/sum(dt_cm_tr))

#Accuracy on test data
sum(diag(dt_cm_te)/sum(dt_cm_te))

#F1 score
dt_cm_te$byClass[,7]
dt_f1 <- mean(dt_cm_te$byClass[,7])

#---------------------K-nearest neighbors---------------------------------0.9801717 - 0.9259121

knn_train_label <- train$Activity_Label
knn_test_label <- test$Activity_Label

numeri <- sapply(data, is.numeric)

nor <-function(x) (x - min(x))/(max(x) - min(x))   

train_knn <- train
test_knn <- test

train_knn[numeri] <- as.data.frame(lapply(train_knn[numeri], nor))
test_knn[numeri] <- as.data.frame(lapply(test_knn[numeri], nor))
library(class)

knn.1 <-  knn(train_knn, test_knn, knn_train_label, k= 5)


#Confusion matrix on trained data

#Confusion matrix on test data
knn_cm_te <- confusionMatrix(knn.1, test$Activity_Label)

#Accuracy on train data
sum(diag(knn_cm_tr$table)/sum(knn_cm_tr$table))

#Accuracy on test data
sum(diag(knn_cm_te$table)/sum(knn_cm_te$table))

#F1 score
knn_cm_te$byClass[,7]
knn_f1 <- mean(knn_cm_te$byClass[,7])
#--------------------SVM-------------------------------------------------0.9717213 - 0.8615125

svm_model <- svm(Activity_Label~., train)
svm_model

#Confusion matrix on trained data
svm_cm_tr <- confusionMatrix(predict(svm_model, train), train$Activity_Label)

#Confusion matrix on test data
svm_cm_te <- confusionMatrix(predict(svm_model, test), test$Activity_Label)

#Accuracy on train data
sum(diag(svm_cm_tr$table)/sum(svm_cm_tr$table))

#Accuracy on test data
sum(diag(svm_cm_te$table)/sum(svm_cm_te$table))

#F1
svm_cm_te$byClass[,7]
svm_f1 <- mean(svm_cm_te$byClass[,7])

#--------------------Neural Network--------------------------------------0.9628053 - 0.9327

n_data <- data
for (i in 1:8){
  if (i==5) {
    next
  }
  n_data[ ,i] <- (n_data[ ,i] - min(n_data[ ,i]))/(max(n_data[ ,i]) - min(n_data[ ,i]))
}
n_data <- as.matrix(n_data)
dimnames(n_data) <- NULL

nn_train <- n_data[ind ==1, 1:8]
nn_test <- n_data[ind ==2, 1:8]
nn_train_y <- n_data[ind ==1, 9]
nn_test_y <- n_data[ind ==2, 9]

nn_train_y_tr <- to_categorical(nn_train_y) #tr - transformed
nn_test_y_tr <- to_categorical(nn_test_y) 

nn_model <- keras_model_sequential()
nn_model %>% 
  layer_dense(units = 42, activation = 'relu', input_shape = c(8)) %>%
  layer_dropout(0.4) %>%
  layer_dense(units = 20, activation = 'relu') %>%
  layer_dropout(0.2) %>%
  layer_dense(units = 4, activation = 'softmax')

nn_model %>% compile(loss = 'categorical_crossentropy',
                     optimizer = 'adam',
                     metrics = 'accuracy')

history <- nn_model %>% 
  fit(nn_train,
      nn_train_y_tr,
      epoch = 100,
      batch_size = 32,
      validation_split = 0.2,
      class_weight = list("0" = 3.1, "1" = 10.5, "2"= 1, "3" = 22.5))

#Confusion matrix on train data
nn_cm_tr <- table(Predicted = nn_model %>% predict_classes(nn_train), Actual = nn_train_y)

#Confusion matrix on test data
nn_cm_te <- table(Predicted = nn_model %>% predict_classes(nn_test), Actual = nn_test_y)

#Accuracy on test data
nn_model %>% evaluate(nn_test, nn_test_y_tr)

nn_f1 <- mean(0.932746, 0.8983462, 0.9975484, 0.6235122)


#--------------------XGBoost---------------------------------------------

xgb_train <- sparse.model.matrix(Activity_Label ~ .-1, data = train)
xgb_train_y <- train[,9]
xgb_train_y <- as.numeric(as.character(xgb_train_y))
xgb_train_matrix <- xgb.DMatrix(data = as.matrix(xgb_train), label = xgb_train_y)

xgb_test <- sparse.model.matrix(Activity_Label ~ .-1, data = test)
xgb_test_y <- test[,9]
xgb_test_y <- as.numeric(as.character(xgb_test_y))
xgb_test_matrix <- xgb.DMatrix(data = as.matrix(xgb_test), label = xgb_test_y)

#parameters
class_num <- length(unique(xgb_train_y))
xgb_parms <- list("objective" = "multi:softprob",
                  "eval_metric" = "mlogloss",
                  "num_class" = class_num)
watchlist <- list(train = xgb_train_matrix, test = xgb_test_matrix)

xgb_model <- xgb.train(params = xgb_parms,
                       data= xgb_train_matrix,
                       nrounds = 100,
                       watchlist = watchlist)

#e <- data.frame(bst_model$evaluation_log)
#plot(e$iter, e$train_mlogloss, col = "blue")
#lines(e$iter, e$test_mlogloss, col = "red")
#xgb.importance(colnames(train_matrix),bst_model)

xgb_pred <- matrix(predict(xgb_model, newdata = xgb_test_matrix),
                   nrow = class_num,
                   ncol = length(predict(xgb_model, newdata = xgb_test_matrix))/class_num) %>%
  t() %>%
  data.frame() %>%
  mutate(label = xgb_test_y, max_prob = max.col(.,"last")-1)

#Confusion matrix on test data
xgb_cm_te <- table(Prediction = xgb_pred$max_prob, Actual = xgb_pred$label)

#Accuracy on test data
sum(diag(xgb_cm_te)/sum(xgb_cm_te))


f1= (2*p*s) / (p + s)


#F1 score
xgb_f1
#xgb_f1 <- f1(0.9887,0.9742)
#xgb_f2 <- f1(0.9866,0.9795)
#xgb_f3 <- f1(0.9992,0.9991)
#xgb_f4 <- f1(0.8174,0.9376)