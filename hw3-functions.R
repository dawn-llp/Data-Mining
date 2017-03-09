# classification function script
# modified based on lecture slides
print("Dependency packages: ROCR, e1071, rpart, class, ada, data.table, randomForest")

# 1. load prepare data: remove ID，missing values and RISK_Adjustment
load.data <- function(data.file ="http://www.yurulin.com/class/spring2017_datamining/data/audit.csv")
{
  dataset <- read.csv(data.file, header=T, sep=",", stringsAsFactors=T) 
  dataset <-na.omit(dataset)
  dataset$ID = NULL
  dataset$RISK_Adjustment = NULL
  print("load data, remove ID, missing values and RISK_Adjustment")
  return(dataset)
}

# 2. exclude which column, get dummy code of categories, scale numeric variables
# randomForest cannot deal with complex var names, so I need to mask categories name.
prepare.data <- function(dataset, dummies=T, maskDummies=F,scale=T, y=10)
{
  print(dataset[1:3,])
  cat.vars <- character(0)
  num.vars <- character(0)
  response <- dataset[,y]
  df <- dataset[,-y]
  cat.vars <- c(cat.vars,names(df)[sapply(df,is.factor)])
  num.vars <- c(num.vars,names(df)[sapply(df,is.numeric)])
  catvars <- df[,cat.vars]
  numvars <- df[,num.vars]
  if(dummies)
  {
    cat("encode ",ncol(catvars),"categories into dummies...",'\n')
    catvars <- model.matrix(~.-1, data=catvars)
  }

  if(maskDummies)
  {
    cat("rename ",ncol(catvars),"dummies' name...",'\n')
    n = ncol(catvars)
    colnames(catvars)[1:n]=paste("cat", 1:n, sep="")
  }
  
  if(scale)
  {
    cat("standardize ",ncol(numvars)," numeric variables 0-1...",'\n')
    numvars = as.matrix(numvars)
    numvars = apply(numvars, MARGIN = 2, FUN = function(X) (X - min(X))/diff(range(X)))
  }
  cat("rename response as y, combine response, categorical variables and numeric 
      variables",'\n','\n')

  response=as.factor(response)
  dataset <- data.frame(response,catvars,numvars)
  colnames(dataset)[1] <- "y"
  print(dataset[1:3,])
  return(dataset)
}

# 3. my.classifer
my.classifier <- function(dataset , cl.name='lr', do.cv=F, nn=3,kopt="linear",
                          gamma=0.01,cost=1, cp=0.01, cutoff=0.5, kfold=10,verbose=F) 
{
  
  if (do.cv) 
  { # classifier = k.fold.cv(dataset, cl.name)
    classifier = k.fold.cv(dataset,cl.name, nn, kopt,gamma,cost, cp, cutoff,kfold,verbose)
   }
  else 
  {  
    classifier = pre.test(dataset,cl.name, nn,kopt,gamma,cost,cp, cutoff,verbose)
  }
  
  return(classifier)
}

#4. simple train/validation 
pre.test <- function(dataset , cl.name, nn,kopt,gamma,cost, cp, cutoff,verbose,
                     ratio=0.6 ) {
  ## Let's use 3/4 random sample as training and remaining as testing
  ## by default use 0.5 as cut-off
  n.obs = nrow(dataset) # no. of observations in dataset
  n.train = floor(n.obs*ratio) # sometimes will have error message
  #s = sample(n.obs)
  train.idx = sample(1:n.obs, size=n.train)
  #test.idx = which(s %% 4 == 3 )
  train.set = dataset[train.idx,]
  test.set = dataset[-train.idx,]
  cat('pre-test',cl.name,':',
      '#training:',nrow(train.set),
      '#testing',nrow(test.set),'\n')
  
  prob = do.classification(train.set, test.set, cl.name,nn,kopt,gamma,cost,cp,verbose)
  
  actual = test.set$y
  matMeasure = MatrixMeasures (prob, actual, cutoff)
  perfMeasure = performanceMeasures (prob,actual)
  ROC = perfMeasure$ROC
  perfMeasure = perfMeasure$perfMeasures
  #plot(perfMeasure$ROC)
  return(list(matMeasure=matMeasure,perfMeasure=perfMeasure,ROC=ROC))
}

# 5.  kfold cross validation
k.fold.cv <- function(dataset , cl.name,nn,kopt,gamma,cost,cp, cutoff, kfold=10,verbose)
{
  ## default: 10-fold CV, cut-off 0.5
  n.obs <- nrow(dataset) # no. of observations
  s = sample(n.obs)
  probs = NULL
  actuals = NULL
  for (i in 1:kfold) {
    test.idx = which(s %% kfold == (i-1) ) # use modular operator
    train.set = dataset[-test.idx ,]
    test.set = dataset[test.idx ,]
    cat(kfold,'-fold CV run',i,cl.name ,':',
        '#training:',nrow(train.set),
        '#testing ',nrow(test.set),'\n')
    prob = do.classification(train.set, test.set, cl.name,nn,kopt,gamma,cost,cp,verbose)
    actual = test.set$y
    probs = c(probs ,prob)
    actuals = c(actuals ,actual)
  }
  matMeasure = MatrixMeasures (probs, actuals, cutoff)
  perfMeasure = performanceMeasures (probs,actuals)
  ROC = perfMeasure$ROC
  perfMeasure = perfMeasure$perfMeasures
  #plot(perfMeasure$ROC)
  return(list(matMeasure=matMeasure,perfMeasure=perfMeasure,ROC=ROC))
}

# 6. basic modeling component
do.classification <- function(train.set, test.set, cl.name, 
                              nn, kopt,gamma,cost,cp, verbose=F) {
  ## note: to plot ROC later , we want the raw
  ## probabilities , not binary decisions
  switch(cl.name,
         knn = { # here we test k=3; you should evaluate different k's
           #library(class)
           model = knn(train.set[,-1], test.set[,-1], cl=train.set[,1], k = nn, prob=T)
           prob = attr(model,"prob") # this prob is for winning class
           model.result = as.character(model)
           model.result = as.numeric(model.result)
           # library(data.table)
           result=data.table(model.result,prob)
           result=result[model.result==0, prob:=1-prob]
           prob = result$prob
           prob
         },
         
         lr = { # logistic regression
           model = glm(y~., family=binomial , data=train.set)
           if(verbose)
           {
             print(summary(model))
           }
           prob = predict.glm(model, newdata=test.set, type = "response")
           prob
         },
         
         nb = { # naiveBayes
           #library(e1071)
           model = naiveBayes(y~., data=train.set)
           if(verbose)
           {
             print(model)
           }
           result = predict(model, newdata=test.set, type="raw")
           prob = result[,2]/rowSums(result)
           prob
         },
         
         tree = { # tree
           #library(rpart)
           model = rpart(y~., data=train.set, cp=cp, method="class" ,maxdepth=10)
           result = predict(model, newdata=test.set)
           prob =result[,which(colnames(result)==1)]/rowSums(result)
           if (verbose) {
             printcp(model) # print the cross-validation results
             plotcp(model) # visualize the cross-validation results
             ## plot the tree
             plot(model, uniform=TRUE, main="Classification Tree")
             text(model, use.n=TRUE, all=TRUE, cex=.8)
           }           
           prob
         },
         
         svm = { # svm
          # library(e1071)
           model = svm(y~., data=train.set,kernel=kopt,gamma=gamma,cost=cost,probability=T)
           result = predict(model, newdata=test.set,probability=T)
           prob = attr(result,"probabilities")
           prob = prob[,which(colnames(prob)==1)]/rowSums(prob)
           if(verbose)
           {
             print(summary(model))
           }
           prob
         },
         
         adaBoost = { # Boosting trees
           #library(ada)
           model = ada(y~., data=train.set, verbose=F)
           result = predict(model, newdata = test.set,type="probs")
           prob = result[,2]/rowSums(result)
           if(verbose)
           {
             print(model)
             varplot(model)
           }
           prob
         },
         
         randomForest = { # Bagging variation
           #library(randomForest)
           model = randomForest(y~.,data=train.set, importance=T,ntree=200)
           result = predict(model, newdata = test.set,type="prob")
           prob = result[,2]/rowSums(result)
           if(verbose)
           {
             print(model)
             varImpPlot(model)
           }
           prob
         }
         
  )
}

         


# 7.  get other measures by using 'performance'
get.measure <- function(pred, measure.name='auc') {
  perf = performance(pred,measure.name)
  m <- unlist(slot(perf, "y.values"))
  m
}

performanceMeasures <- function(prob,actual)
{
  pred = prediction(prob ,actual)
  accuracy = mean(get.measure(pred, 'acc'),na.rm=T)
  precision = mean(get.measure(pred, 'prec'),na.rm=T)
  recall = mean(get.measure(pred, 'rec'),na.rm=T)
  fscore = mean(get.measure(pred, 'f'),na.rm=T)
  print('ROCR performance function calculated measures:')
  cat('accuracy=',accuracy,'precision=',precision,'recall=',recall,'f-score=',fscore,'\n')
  auc = get.measure(pred, 'auc')
  cat('auc=',auc,'\n')
  ROC = performance(pred, "tpr", "fpr")
  perfMeasures = data.frame(accuracy,precision,recall,fscore,auc)
  return(a =list(perfMeasures=perfMeasures, ROC=ROC) )
}

# 8. Calculate measures with specific cutoff
MatrixMeasures <- function(prob, actual, cutoff)
{
  predicted = as.numeric(prob > cutoff)
  conf.matrix = table(factor(actual ,levels=c(0,1)) ,
                      factor(predicted ,levels=c(0,1)))
  accuracy = (conf.matrix[1,1]+
                conf.matrix[2,2]) / sum(conf.matrix)
  
  # Add other measures: precision, recall, f-means, and auc
  precision = conf.matrix[2,2]/(conf.matrix[2,2]+conf.matrix[1,2])
  recall = conf.matrix[2,2]/(conf.matrix[2,2]+conf.matrix[2,1])
  fscore = (2*(precision)*(recall))/(precision+recall)
  
  print('Confusion matrix calculated measures:')
  cat('accuracy=',accuracy,'precision=',precision,'recall=',recall,'f-score=',fscore,'\n',
      'cutoff=',cutoff,'\n')
  
  matMeasures = data.frame(accuracy,precision,recall,fscore)
  return(matMeasures)
}

# tsvm = { # tuned svm
#   tuned <- tune.svm(y~., data = train.set, kernel=kopt, 
#                     gamma = 10^(-6:-1), cost = 10^(-1:1))
#   gamma = tuned[['best.parameters']]$gamma
#   cost = tuned[['best.parameters']]$cost
#   model = svm(y~., data = train.set, kernel=kopt, gamma=gamma,
#               cost=cost,probability=T)                        
#   prob = predict(model, newdata=test.set, probability=T)
#   if(verbose)
#   {
#     print(tuned)
#     print(summary(model))
#   }
#   prob
# }

# ptree = { # pruned tree
#   # library(rpart)
#   model1 = rpart(y~., data=train.set, method="class" )
#   model2 = prune(model1, cp=model1$cptable[which.min(model1$cptable[,"xerror"]),"CP"],
#                  xval=10)
#   result = predict(model2, newdata=test.set)
#   prob =result[,which(colnames(result)==1)]/rowSums(result)
#   if (verbose) {
#     printcp(model2) # detailed summary of splits
#     ## plot the tree
#     plot(model2, uniform=TRUE, main="Pruned Classification Tree")
#     text(model2, use.n=TRUE, all=TRUE, cex=.8)
#   }           
#   
#   prob
# },