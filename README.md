
# Introduction
This dataset contains the motion data of 14 healthy older aged between 66 and 86 years old, performed broadly scripted activities using a batteryless, wearable sensor on top of their clothing at sternum level.  

## Preparations

Libraries that will be used for project:

```r
library(data.table)
library(dplyr)
library(caret)
library(MLmetrics)
library(nnet)
library(randomForest)
library(rpart)
library(class)
library(e1071)
library(xgboost)
library(Matrix)
library(magrittr)
library(keras)
library(tensorflow)
```

### Reading data


```r
setwd("S2_Dataset/")
reading_data_1 <- function(){
  path <- "C:\\Users\\Lukas\\Desktop\\R2/Activity_recognition/S2_Dataset/"
  files <- list.files(path, "p")
  l <- lapply(files, fread, header = FALSE, sep = ",", stringsAsFactors = FALSE, showProgress = FALSE)
  data <- do.call(rbind, l)
  return(data)
}
data <- as.data.frame(reading_data_1())
```

### Exploring data

```r
cat("Number of observations in dataset:", dim(data)[1],", number of features:", dim(data)[2])
```
Number of observations in dataset: 75128 , number of features: 9

```r
str(data)
```

```
## 'data.frame':	75128 obs. of  9 variables:
##  $ V1: num  0 0.5 1.5 1.75 2.5 3.25 4 5 5.5 6 ...
##  $ V2: num  0.272 0.272 0.448 0.448 0.342 ...
##  $ V3: num  1.008 1.008 0.916 0.916 0.962 ...
##  $ V4: num  -0.0821 -0.0821 -0.0137 -0.0137 -0.0593 ...
##  $ V5: int  1 1 1 1 1 4 1 1 1 1 ...
##  $ V6: num  -63.5 -63 -63.5 -63 -63.5 -56.5 -63.5 -64 -64.5 -66 ...
##  $ V7: num  2.43 4.74 3.03 2.04 5.89 ...
##  $ V8: num  924 922 924 921 920 ...
##  $ V9: int  1 1 1 1 1 1 1 1 1 1 ...
```
```r
summary(data)
```

```
##        V1               V2                V3                  V4          
##  Min.   :   0.0   Min.   :-0.7481   Min.   :-0.553490   Min.   :-1.33640  
##  1st Qu.: 121.2   1st Qu.: 0.3424   1st Qu.:-0.002297   1st Qu.:-0.18473  
##  Median : 250.7   Median : 0.6824   Median : 0.215880   Median :-0.07070  
##  Mean   : 299.1   Mean   : 0.7142   Mean   : 0.345199   Mean   :-0.21748  
##  3rd Qu.: 402.5   3rd Qu.: 1.1045   3rd Qu.: 0.858940   3rd Qu.: 0.03193  
##  Max.   :1739.4   Max.   : 1.5032   Max.   : 2.030200   Max.   : 1.21780  
##        V5             V6               V7              V8       
##  Min.   :1.00   Min.   :-72.00   Min.   :0.000   Min.   :920.2  
##  1st Qu.:1.00   1st Qu.:-62.00   1st Qu.:1.032   1st Qu.:921.2  
##  Median :3.00   Median :-58.00   Median :2.767   Median :922.8  
##  Mean   :2.41   Mean   :-58.28   Mean   :3.157   Mean   :922.7  
##  3rd Qu.:3.00   3rd Qu.:-56.00   3rd Qu.:5.359   3rd Qu.:924.2  
##  Max.   :4.00   Max.   :-38.50   Max.   :6.282   Max.   :925.8  
##        V9       
##  Min.   :1.000  
##  1st Qu.:2.000  
##  Median :3.000  
##  Mean   :2.528  
##  3rd Qu.:3.000  
##  Max.   :4.000
```
Missing values

```r
cat("Number of missing values in dataset:",  sum(is.na(data)))
```
```
Number of missing values in dataset: 0
```

### Preparing data

```r
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
```

## Machine learning


```r
set.seed(1234)
ind <- sample(2, nrow(data), replace = T, prob = c(0.8, 0.2))
train <- data[ind == 1, ]
test <- data[ind==2, ]
```

### Multinomial logistic regression


```r
mlg_model <- multinom(Activity_Label ~.,data = train)
```

```
## # weights:  48 (33 variable)
## initial  value 83314.904809 
## iter  10 value 46736.621665
## iter  20 value 31291.781527
## iter  30 value 14826.022383
## iter  40 value 8622.197593
## iter  50 value 8363.243477
## iter  60 value 8274.474744
## iter  70 value 8259.677072
## iter  80 value 8258.739796
## iter  90 value 8253.779351
## iter 100 value 8253.114867
## final  value 8253.114867 
## stopped after 100 iterations
```

```r
#Confusion matrix on test data
mlg_cm_te <- confusionMatrix(predict(mlg_model,test), test$Activity_Label)
mlg_cm_te
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction     0     1     2     3
##          0  3135   117     6   264
##          1   110   834     3    40
##          2    34     0 10310     3
##          3     1    21     0   151
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9601          
##                  95% CI : (0.9569, 0.9632)
##     No Information Rate : 0.6866          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9155          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: 0 Class: 1 Class: 2 Class: 3
## Sensitivity            0.9558  0.85802   0.9991  0.32969
## Specificity            0.9671  0.98912   0.9921  0.99849
## Pos Pred Value         0.8901  0.84498   0.9964  0.87283
## Neg Pred Value         0.9874  0.99017   0.9981  0.97933
## Prevalence             0.2182  0.06467   0.6866  0.03047
## Detection Rate         0.2086  0.05549   0.6860  0.01005
## Detection Prevalence   0.2343  0.06567   0.6885  0.01151
## Balanced Accuracy      0.9614  0.92357   0.9956  0.66409
```

```r
#Accuracy on test data
cat("Accuracy on test data:",  sum(diag(mlg_cm_te$table)/sum(mlg_cm_te$table)))
```

```
## Accuracy on test data: 0.9601437
```

```r
mlg_f1 <- mean(mlg_cm_te$byClass[,7])

cat("F1 score:",  mlg_f1)
```

```
## F1 score: 0.8124055
```

### randomForest


```r
rf_model <- randomForest(Activity_Label~.,data = train)

#Confusion matrix on test data
rf_cm_te <- confusionMatrix(predict(rf_model,test), test$Activity_Label)
rf_cm_te
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction     0     1     2     3
##          0  3249     7     7    87
##          1     8   965     0    14
##          2    16     0 10312     1
##          3     7     0     0   356
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9902          
##                  95% CI : (0.9885, 0.9917)
##     No Information Rate : 0.6866          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9794          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: 0 Class: 1 Class: 2 Class: 3
## Sensitivity            0.9905  0.99280   0.9993  0.77729
## Specificity            0.9914  0.99843   0.9964  0.99952
## Pos Pred Value         0.9699  0.97771   0.9984  0.98072
## Neg Pred Value         0.9973  0.99950   0.9985  0.99305
## Prevalence             0.2182  0.06467   0.6866  0.03047
## Detection Rate         0.2162  0.06421   0.6861  0.02369
## Detection Prevalence   0.2229  0.06567   0.6873  0.02415
## Balanced Accuracy      0.9910  0.99562   0.9979  0.88841
```

```r
#Accuracy on test data
cat("Accuracy on test data:",sum(diag(rf_cm_te$table)/sum(rf_cm_te$table)))
```

```
## Accuracy on test data: 0.9902189
```

```r
#F1 score
rf_f1 <- mean(rf_cm_te$byClass[,7])
cat(" F1 score:",rf_f1)
```

```
##  F1 score: 0.9578399
```

### Decision tree

```r
dt_model <- rpart(Activity_Label ~.,data = train)

#Confusion matrix on test data
dt_cm_te <- confusionMatrix(predict(dt_model, test, type="class"), test$Activity_Label)
dt_cm_te
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction     0     1     2     3
##          0  3097   196     8   280
##          1   100   776     1    36
##          2    82     0 10309     7
##          3     1     0     1   135
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9526         
##                  95% CI : (0.9491, 0.956)
##     No Information Rate : 0.6866         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.8989         
##                                          
##  Mcnemar's Test P-Value : < 2.2e-16      
## 
## Statistics by Class:
## 
##                      Class: 0 Class: 1 Class: 2 Class: 3
## Sensitivity            0.9442  0.79835   0.9990 0.294760
## Specificity            0.9588  0.99025   0.9811 0.999863
## Pos Pred Value         0.8648  0.84995   0.9914 0.985401
## Neg Pred Value         0.9840  0.98612   0.9978 0.978311
## Prevalence             0.2182  0.06467   0.6866 0.030474
## Detection Rate         0.2061  0.05163   0.6859 0.008983
## Detection Prevalence   0.2383  0.06075   0.6919 0.009116
## Balanced Accuracy      0.9515  0.89430   0.9901 0.647311
```

```r
#Accuracy on test data
cat(" Accuracy on test data:", sum(diag(dt_cm_te$table)/sum(dt_cm_te$table)))
```

```
##  Accuracy on test data: 0.9526249
```

```r
#F1 score
dt_f1 <- mean(dt_cm_te$byClass[,7])
cat(" F1 score:", dt_f1)
```

```
##  F1 score: 0.7937822
```

### K-nearest neighbors

```r
# 
# knn_model <- caret::train(Activity_Label ~.,
#                    data = train,
#                    method = 'knn',
#                    tuneLength = 20,
#                    preProc = c("center", "scale")) 
# 
# #Confusion matrix on test data
# knn_cm_te <- confusionMatrix(predict(knn_model, test), test$Activity_Label)
# knn_cm_te
# 
# #Accuracy on test data
# sum(diag(knn_cm_te$table)/sum(knn_cm_te$table))
# 
# #F1 score
# knn_f1 <- mean(knn_cm_te$byClass[,7])
# knn_f1
```

### SVM


```r
svm_model <- svm(Activity_Label~., train)
svm_model
```

```
## 
## Call:
## svm(formula = Activity_Label ~ ., data = train)
## 
## 
## Parameters:
##    SVM-Type:  C-classification 
##  SVM-Kernel:  radial 
##        cost:  1 
## 
## Number of Support Vectors:  4665
```

```r
#Confusion matrix on test data
svm_cm_te <- confusionMatrix(predict(svm_model, test), test$Activity_Label)
svm_cm_te
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction     0     1     2     3
##          0  3176    42     9   229
##          1    71   930     0    36
##          2    30     0 10310     5
##          3     3     0     0   188
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9717          
##                  95% CI : (0.9689, 0.9743)
##     No Information Rate : 0.6866          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9401          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: 0 Class: 1 Class: 2 Class: 3
## Sensitivity            0.9683  0.95679   0.9991  0.41048
## Specificity            0.9762  0.99239   0.9926  0.99979
## Pos Pred Value         0.9190  0.89682   0.9966  0.98429
## Neg Pred Value         0.9910  0.99700   0.9981  0.98180
## Prevalence             0.2182  0.06467   0.6866  0.03047
## Detection Rate         0.2113  0.06188   0.6860  0.01251
## Detection Prevalence   0.2300  0.06900   0.6883  0.01271
## Balanced Accuracy      0.9722  0.97459   0.9958  0.70514
```

```r
#Accuracy on test data
cat(" Accuracy on test data:", sum(diag(svm_cm_te$table)/sum(svm_cm_te$table)))
```

```
##  Accuracy on test data: 0.9717213
```

```r
#F1
svm_f1 <- mean(svm_cm_te$byClass[,7])
cat(" F1 score:", svm_f1)
```

```
##  F1 score: 0.8615125
```

### Neural Network


```r
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
      epoch = 30,
      batch_size = 32,
      validation_split = 0.2,
      class_weight = list("0" = 3.1, "1" = 10.5, "2"= 1, "3" = 22.5))

#Confusion matrix on train data
nn_cm_tr <- table(Predicted = nn_model %>% predict_classes(nn_train), Actual = nn_train_y)

#Confusion matrix on test data
nn_cm_te <- table(Predicted = nn_model %>% predict_classes(nn_test), Actual = nn_test_y)

#Accuracy on test data
nn_model %>% evaluate(nn_test, nn_test_y_tr)
```

```
## $loss
## [1] 0.1771907
## 
## $acc
## [1] 0.9639364
```

```r
nn_f1 <- mean(0.932746, 0.8983462, 0.9975484, 0.6235122)
nn_f1
```

```
## [1] 0.932746
```

### XGBoost


```r
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
```

```
## [1]	train-mlogloss:0.883199	test-mlogloss:0.883841 
## [2]	train-mlogloss:0.621997	test-mlogloss:0.623183 
## [3]	train-mlogloss:0.456996	test-mlogloss:0.458734 
## [4]	train-mlogloss:0.345408	test-mlogloss:0.347618 
## [5]	train-mlogloss:0.267178	test-mlogloss:0.269954 
## [6]	train-mlogloss:0.210588	test-mlogloss:0.214099 
## [7]	train-mlogloss:0.169574	test-mlogloss:0.173546 
## [8]	train-mlogloss:0.139327	test-mlogloss:0.143865 
## [9]	train-mlogloss:0.116865	test-mlogloss:0.122002 
## [10]	train-mlogloss:0.099209	test-mlogloss:0.105146 
## [11]	train-mlogloss:0.086273	test-mlogloss:0.093004 
## [12]	train-mlogloss:0.076259	test-mlogloss:0.083534 
## [13]	train-mlogloss:0.068432	test-mlogloss:0.076023 
## [14]	train-mlogloss:0.062296	test-mlogloss:0.070495 
## [15]	train-mlogloss:0.057569	test-mlogloss:0.066026 
## [16]	train-mlogloss:0.053996	test-mlogloss:0.062687 
## [17]	train-mlogloss:0.051107	test-mlogloss:0.059949 
## [18]	train-mlogloss:0.048391	test-mlogloss:0.057687 
## [19]	train-mlogloss:0.046670	test-mlogloss:0.056252 
## [20]	train-mlogloss:0.045127	test-mlogloss:0.054946 
## [21]	train-mlogloss:0.042940	test-mlogloss:0.053158 
## [22]	train-mlogloss:0.041812	test-mlogloss:0.052251 
## [23]	train-mlogloss:0.040607	test-mlogloss:0.051224 
## [24]	train-mlogloss:0.039159	test-mlogloss:0.050124 
## [25]	train-mlogloss:0.038317	test-mlogloss:0.049489 
## [26]	train-mlogloss:0.037346	test-mlogloss:0.048654 
## [27]	train-mlogloss:0.036313	test-mlogloss:0.047898 
## [28]	train-mlogloss:0.035056	test-mlogloss:0.047028 
## [29]	train-mlogloss:0.033770	test-mlogloss:0.046181 
## [30]	train-mlogloss:0.032686	test-mlogloss:0.045394 
## [31]	train-mlogloss:0.032113	test-mlogloss:0.045080 
## [32]	train-mlogloss:0.031477	test-mlogloss:0.044681 
## [33]	train-mlogloss:0.030011	test-mlogloss:0.043685 
## [34]	train-mlogloss:0.029027	test-mlogloss:0.043125 
## [35]	train-mlogloss:0.028208	test-mlogloss:0.042533 
## [36]	train-mlogloss:0.027068	test-mlogloss:0.042093 
## [37]	train-mlogloss:0.026632	test-mlogloss:0.041803 
## [38]	train-mlogloss:0.026143	test-mlogloss:0.041641 
## [39]	train-mlogloss:0.025278	test-mlogloss:0.041061 
## [40]	train-mlogloss:0.024607	test-mlogloss:0.040718 
## [41]	train-mlogloss:0.024099	test-mlogloss:0.040397 
## [42]	train-mlogloss:0.023503	test-mlogloss:0.040128 
## [43]	train-mlogloss:0.022777	test-mlogloss:0.039690 
## [44]	train-mlogloss:0.022323	test-mlogloss:0.039435 
## [45]	train-mlogloss:0.021840	test-mlogloss:0.039120 
## [46]	train-mlogloss:0.021388	test-mlogloss:0.038958 
## [47]	train-mlogloss:0.020969	test-mlogloss:0.038761 
## [48]	train-mlogloss:0.020438	test-mlogloss:0.038377 
## [49]	train-mlogloss:0.019827	test-mlogloss:0.038186 
## [50]	train-mlogloss:0.019269	test-mlogloss:0.037988 
## [51]	train-mlogloss:0.018691	test-mlogloss:0.037671 
## [52]	train-mlogloss:0.017814	test-mlogloss:0.037270 
## [53]	train-mlogloss:0.017603	test-mlogloss:0.037184 
## [54]	train-mlogloss:0.017181	test-mlogloss:0.037076 
## [55]	train-mlogloss:0.016883	test-mlogloss:0.036865 
## [56]	train-mlogloss:0.016453	test-mlogloss:0.036587 
## [57]	train-mlogloss:0.016059	test-mlogloss:0.036530 
## [58]	train-mlogloss:0.015558	test-mlogloss:0.036229 
## [59]	train-mlogloss:0.015185	test-mlogloss:0.035908 
## [60]	train-mlogloss:0.014907	test-mlogloss:0.035748 
## [61]	train-mlogloss:0.014765	test-mlogloss:0.035648 
## [62]	train-mlogloss:0.014561	test-mlogloss:0.035529 
## [63]	train-mlogloss:0.014250	test-mlogloss:0.035460 
## [64]	train-mlogloss:0.013803	test-mlogloss:0.035327 
## [65]	train-mlogloss:0.013497	test-mlogloss:0.035156 
## [66]	train-mlogloss:0.013162	test-mlogloss:0.034950 
## [67]	train-mlogloss:0.012952	test-mlogloss:0.034854 
## [68]	train-mlogloss:0.012626	test-mlogloss:0.034781 
## [69]	train-mlogloss:0.012425	test-mlogloss:0.034720 
## [70]	train-mlogloss:0.012243	test-mlogloss:0.034672 
## [71]	train-mlogloss:0.012082	test-mlogloss:0.034609 
## [72]	train-mlogloss:0.011691	test-mlogloss:0.034388 
## [73]	train-mlogloss:0.011437	test-mlogloss:0.034282 
## [74]	train-mlogloss:0.011194	test-mlogloss:0.034162 
## [75]	train-mlogloss:0.010945	test-mlogloss:0.034068 
## [76]	train-mlogloss:0.010733	test-mlogloss:0.033919 
## [77]	train-mlogloss:0.010591	test-mlogloss:0.033878 
## [78]	train-mlogloss:0.010446	test-mlogloss:0.033887 
## [79]	train-mlogloss:0.010225	test-mlogloss:0.033754 
## [80]	train-mlogloss:0.010062	test-mlogloss:0.033706 
## [81]	train-mlogloss:0.009848	test-mlogloss:0.033634 
## [82]	train-mlogloss:0.009655	test-mlogloss:0.033492 
## [83]	train-mlogloss:0.009446	test-mlogloss:0.033386 
## [84]	train-mlogloss:0.009302	test-mlogloss:0.033375 
## [85]	train-mlogloss:0.009140	test-mlogloss:0.033275 
## [86]	train-mlogloss:0.008847	test-mlogloss:0.033153 
## [87]	train-mlogloss:0.008591	test-mlogloss:0.033074 
## [88]	train-mlogloss:0.008444	test-mlogloss:0.033062 
## [89]	train-mlogloss:0.008358	test-mlogloss:0.033095 
## [90]	train-mlogloss:0.008232	test-mlogloss:0.033013 
## [91]	train-mlogloss:0.008138	test-mlogloss:0.033023 
## [92]	train-mlogloss:0.008087	test-mlogloss:0.033034 
## [93]	train-mlogloss:0.007861	test-mlogloss:0.033019 
## [94]	train-mlogloss:0.007731	test-mlogloss:0.033028 
## [95]	train-mlogloss:0.007669	test-mlogloss:0.032997 
## [96]	train-mlogloss:0.007505	test-mlogloss:0.032878 
## [97]	train-mlogloss:0.007424	test-mlogloss:0.032871 
## [98]	train-mlogloss:0.007335	test-mlogloss:0.032872 
## [99]	train-mlogloss:0.007271	test-mlogloss:0.032836 
## [100]	train-mlogloss:0.007147	test-mlogloss:0.032832
```

```r
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
```

```
## [1] 0.9906847
```

```r
F1_Score(xgb_pred$label, xgb_pred$max_prob)
```

```
## [1] 0.981389
```
