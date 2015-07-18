---
title: "Predicting the Activities for the Weight Lifting Exercise"
author: "Kusum Subedi"
date: "Wednesday, July 15, 2015"
output: html_document
---

#### Synopsis

Nowadays it is possible to collect a large number of data about personal activity relatively inexpensively with the use of devices such as Jawbone Up, Nike FuelBand and Fitbit.

In this project we are using  data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.
They are as follows:

- Class A: exactly according to the specification 

- Class B: throwing the elbows to the front 

- Class C: lifting the dumbbell only halfway 

- Class D: lowering the dumbbell only halfway

- Class E: throwing the hips to the front 

Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes. Participants were supervised by an experienced weight lifter to make sure the execution complied to the manner they were supposed to simulate. The exercises were performed by six male participants aged between 20-28 years, with little weight lifting experience. It was made sure that all participants could easily simulate the mistakes in a safe and controlled manner by using a relatively light dumbbell (1.25kg).


**Dataset:** The test and train data can be obtained from:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

#### These data have been provided by:

http://groupware.les.inf.puc-rio.br/har

First the data has been downloaded. First we just analysing the train data without touching the test data. When the train data is viewed, we can see there are so many predictor variables.The first 7 columns are irrelevant for the analysis.



```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.1.3
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```
## Warning: package 'ggplot2' was built under R version 3.1.3
```

```r
library(kernlab)
```

```
## Warning: package 'kernlab' was built under R version 3.1.3
```

```r
trainingRaw <- read.csv("pml-training.csv")
testingRaw <- read.csv("pml-testing.csv")
```

So the first thing to do is to reduce the number of the predictors. To do so, I chose to delete the columns with too many NAs and blanks.To reduce it further, I deleted the columns with entries that have near zero variance. After this there were just 52 predictors left. In this particular case nearZeroVar did not make any difference but in many cases it is really a good tool to reduce the predictors.


```r
trainingRaw_1 <- trainingRaw[,8:160]
threshold <- dim(trainingRaw)[1]*0.95
trainingRaw_2 <- trainingRaw_1[, !apply(trainingRaw_1, 2,
                                        function(x) sum(is.na(x))> threshold || sum(x=="") > threshold)]

nsv <- nearZeroVar(trainingRaw_2, saveMetrics=TRUE)
training <- trainingRaw_2[ , grep("FALSE", nsv$nzv)]
```


Then I used this subset data to model fit.
 

```r
modelFit <- train(training$classe ~., method="qda",preProcess= c("center", "scale"), data=training)
```

```
## Loading required package: MASS
```

```r
modelFit
```

```
## Quadratic Discriminant Analysis 
## 
## 19622 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## Pre-processing: centered, scaled 
## Resampling: Bootstrapped (25 reps) 
## 
## Summary of sample sizes: 19622, 19622, 19622, 19622, 19622, 19622, ... 
## 
## Resampling results
## 
##   Accuracy   Kappa      Accuracy SD  Kappa SD   
##   0.8929487  0.8647692  0.004409498  0.005523349
## 
## 
```

The acuracy of this model is about 89% which implies the out of sample error is about 11%, not bad.

For more accurate result, cross validation is performed.


```r
set.seed(125)
inTrain <- createDataPartition(y = training$classe, p = 0.6, list = FALSE)
cvTrain <- training[inTrain, ]
cvTest <- training[-inTrain, ]
fitCtrl <- trainControl(method = "repeatedcv", number = 10, repeats = 10)

modelFit1 <- train(cvTrain$classe ~ ., data = cvTrain, method = "qda", 
                   preProcess = c("center", "scale") , trControl = fitCtrl)
modelFit1
```

```
## Quadratic Discriminant Analysis 
## 
## 11776 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## Pre-processing: centered, scaled 
## Resampling: Cross-Validated (10 fold, repeated 10 times) 
## 
## Summary of sample sizes: 10599, 10598, 10600, 10598, 10597, 10599, ... 
## 
## Resampling results
## 
##   Accuracy   Kappa      Accuracy SD  Kappa SD  
##   0.8926287  0.8644717  0.008897382  0.01121533
## 
## 
```

```r
testPrediction <- predict(modelFit1, newdata=cvTest)

confusionmatrix <- confusionMatrix(data=testPrediction, reference=cvTest$classe)
confusionmatrix
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2052   67    1    0    0
##          B   96 1260   74    7   31
##          C   50  167 1282  197   50
##          D   28    5    4 1072   32
##          E    6   19    7   10 1329
## 
## Overall Statistics
##                                           
##                Accuracy : 0.8915          
##                  95% CI : (0.8844, 0.8983)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.8631          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9194   0.8300   0.9371   0.8336   0.9216
## Specificity            0.9879   0.9671   0.9284   0.9895   0.9934
## Pos Pred Value         0.9679   0.8583   0.7342   0.9395   0.9694
## Neg Pred Value         0.9686   0.9595   0.9859   0.9681   0.9825
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2615   0.1606   0.1634   0.1366   0.1694
## Detection Prevalence   0.2702   0.1871   0.2225   0.1454   0.1747
## Balanced Accuracy      0.9536   0.8986   0.9328   0.9115   0.9575
```

This result also showed the accuracy is 89%. So the out of sample error can be estimated to be about 11%.

When I used the prediction given by this model on the given test data I got 19 correct out of 20 (95%), which is infact better than estimation.


 
 
