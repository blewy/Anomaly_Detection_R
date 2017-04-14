library(e1071)
library(caret)
library(NLP)
library(tm)
library(h2o)
library(data.table)
library(tidyverse)
library(ggplot2)
library(gridExtra)
library(scatterplot3d)
library(caret)
library(tidyverse)


# ------------  H2O autoencoders  -------------


set.seed(12345)
cl <- h2o.init(
  max_mem_size = "4G",
  nthreads = 10)

data(iris)
df <- iris
dim(df)


preProcValues <- preProcess(iris[,-5], method = c("range"))
#preProcValues <- preProcess(iris[,-5], method = c("center", "scale"))
irisTransformed <- predict(preProcValues, iris)
#irisTransformed <- iris

inTrain<-createDataPartition(1:nrow(irisTransformed),p=0.75,list=FALSE)

h2oIris <- as.h2o(
  irisTransformed,
  destination_frame = "h2oIris")

h2oIris.train <- h2oIris[inTrain, -5]
dim(h2oIris.train)
h2oIris.test <- h2oIris[-inTrain, -5]
xnames <- colnames(h2oIris.train)
dim(h2oIris.test)
folds <- createFolds(1:dim(h2oIris.train)[1], k = 5)


## create parameters to try
hyperparams <- list(
  list(# M1
    hidden = c(2),
    input_dr = c(0),
    hidden_dr = c(0),
    activation_dr='TanhWithDropout',
    l1_dr=0,
    l2_dr=0),
  list(# M2
    hidden = c(2),
    input_dr = c(.2),
    hidden_dr = c(0),
    activation_dr='TanhWithDropout',
    l1_dr=0,
    l2_dr=0),
  list(# M3
    hidden = c(5),
    input_dr = c(.2),
    hidden_dr = c(0),
    activation_dr='TanhWithDropout',
    l1_dr=0,
    l2_dr=0),
  list(# M4
    hidden = c(5),
    input_dr = c(.2),
    hidden_dr = c(.5),
    activation_dr='TanhWithDropout',
    l1_dr=0,
    l2_dr=0),
  list(# M5
    hidden = c(2),
    input_dr = c(0),
    hidden_dr = c(0),
    activation_dr='RectifierWithDropout',
    l1_dr=0,
    l2_dr=0),
  list(# M6
    hidden = c(2),
    input_dr = c(.2),
    hidden_dr = c(0),
    activation_dr='RectifierWithDropout',
    l1_dr=0,
    l2_dr=0),
  list(# M7
    hidden = c(5),
    input_dr = c(.2),
    hidden_dr = c(0),
    activation_dr='RectifierWithDropout',
    l1_dr=0,
    l2_dr=0),
  list(# M8
    hidden = c(5),
    input_dr = c(.2),
    hidden_dr = c(.5),
    activation_dr='RectifierWithDropout',
    l1_dr=0,
    l2_dr=0)
 
  )


fm <- lapply(hyperparams, function(v) {
  lapply(folds, function(i) {
    h2o.deeplearning(
      x = xnames,
      training_frame = h2oIris.train[-i, ],
      validation_frame = h2oIris.train[i, ],
      activation =  v$activation_dr,
      autoencoder = TRUE,
      hidden = v$hidden,
      epochs = 30,
      sparsity_beta = 0,
      input_dropout_ratio = v$input_dr,
      hidden_dropout_ratios = v$hidden_dr,
      l1 = v$l1_dr,
      l2 = v$l2_dr
    )
  })
})


fm.res <- lapply(fm, function(m) {
  sapply(m, h2o.mse, valid = TRUE)
})

fm.res <- data.table(
  Model = rep(paste0("M", 1:8), each = 5),
  MSE = unlist(fm.res))


head(fm.res)

p.erate <- ggplot(fm.res, aes(reorder(Model,MSE,FUN=median), MSE)) +
  geom_boxplot() +
  stat_summary(fun.y = mean, geom = "point", colour = "red") +
  theme_classic()
p.erate

fm.res[, .(Mean_MSE = mean(MSE)), by = Model][order(Mean_MSE)]

best<-3

fm.final <- h2o.deeplearning(
  x = xnames,
  training_frame = h2oIris.train,
  validation_frame = h2oIris.test,
  activation = hyperparams[[best]]$activation_dr,
  autoencoder = TRUE,
  hidden = hyperparams[[best]]$hidden,
  epochs = 30,
  sparsity_beta = 0,
  input_dropout_ratio = hyperparams[[best]]$input_dr,
  hidden_dropout_ratios = hyperparams[[best]]$hidden_dr,
  l1 = hyperparams[[best]]$l1_dr,
  l2 = hyperparams[[best]]$l2_dr
)



error2a <-as.data.frame(h2o.anomaly(fm.final, h2oIris))

ggplot(data=error2a, aes(Reconstruction.MSE)) +  geom_density()

data<-cbind(df,error2a)
#outlier threshold
thresh <- 0.95
Percentile = quantile(error2a$Reconstruction.MSE, probs = thresh)

data$error <- (data$Reconstruction.MSE>Percentile)


Auto_Encoder.plot1 <- ggplot(data=data, aes(x=Sepal.Length, y=Sepal.Width,color=Species , shape=Species, size=(error)*0.1)) + geom_point()+  ggtitle("H2O Auto Encoder")
Auto_Encoder.plot1

Auto_Encoder.plot2 <- ggplot(data=data, aes(x=Petal.Length, y=Petal.Width,color=Species , shape=Species, size=(error)*0.1)) + geom_point()+  ggtitle("H2O Auto Encoder")
Auto_Encoder.plot2
 
# ------------  svm 1-class -------------


# • you want that every datapoint is within (on the + side) or at the border of the hyperplane then set nu <= 1/n.

# • you want to go for the mode set nu=1 and all datapoints are outside (on the - side) of the hyperplane

# • you expect your sample to contain 10% outliers: nu = 1–0.1 = 0.9, hence, 90% of your datapoints will lie inside (on the + side) the hyperplane
set.seed(12345)

svm.model<-svm(df[,-5],y=NULL,
               type='one-classification',
               nu=0.05,
               scale=TRUE,
               kernel="radial")
#nu can be seen as outlier threshold

svm.predtrain<-predict(svm.model,df[,-5])


data2<-cbind(data,svm.predtrain)

SVM_One_Class.plot1<- ggplot(data=data, aes(x=Sepal.Length, y=Sepal.Width,color=Species , shape=Species, size=(1-svm.predtrain)*0.1)) + geom_point() +  ggtitle("SVM One-Class")
SVM_One_Class.plot1

SVM_One_Class.plot2<- ggplot(data=data, aes(x=Petal.Length, y=Petal.Width,color=Species , shape=Species, size=(1-svm.predtrain)*0.1)) + geom_point() +  ggtitle("SVM One-Class")
SVM_One_Class.plot2


# remove species from the data to cluster
#iris2 <- iris[,1:4]
#


# ------------  K-means clustering -------------

set.seed(12345)
data(iris)
df <- iris
dim(df)


kmeans.result <- kmeans(df[,-5], centers=3,iter.max = 1000, nstart = 50)
# cluster centers
kmeans.result$centers
# calculate distances between objects and cluster centers
centers <- kmeans.result$centers[kmeans.result$cluster, ]
distances <- sqrt(rowSums((df[,-5] - centers)^2))
# pick top 5 largest distances
outliers <- order(distances, decreasing=T)[1:5]
# who are outliers
print(outliers)

# calculate mean distances by cluster:
m <- tapply(distances, kmeans.result$cluster,mean)
# calculate mean distances by cluster:
std <- tapply(distances, kmeans.result$cluster,sd)



# divide each distance by the mean for its cluster:
d <- distances/(m[kmeans.result$cluster])

d[order(d, decreasing=TRUE)][1:5]

# divide each distance by the mean for its cluster:
d <- abs((distances - (m[kmeans.result$cluster])) / (std[kmeans.result$cluster]))

#outlier threshold
Percentile<-0.05

error_kmeans <- d > qnorm((1-Percentile/2), mean = 0, sd = 1, lower.tail = TRUE, log.p = FALSE)
data3<-cbind(data2,error_kmeans)

cluster.plot1 <- ggplot(data=data3, aes(x=Sepal.Length, y=Sepal.Width,color=Species , shape=Species, size=(error_kmeans)*0.1)) + geom_point() +  ggtitle("Cluster Outliers")
cluster.plot1

cluster.plot2 <- ggplot(data=data3, aes(x=Petal.Length, y=Petal.Width,color=Species , shape=Species, size=(error_kmeans)*0.1)) + geom_point() +  ggtitle("Cluster Outliers")
cluster.plot2

#grid.arrange(cluster.plot1,SVM_One_Class.plot1,Auto_Encoder.plot1,cluster.plot2,SVM_One_Class.plot2,Auto_Encoder.plot2, ncol=3,nrow=2)


#---   LOF for anomaly detection --------------

library(Rlof)

#The number of neighbors considered, (parameter n_neighbors) is typically chosen 
#1) greater than the minimum number of objects a cluster has to contain, so that other objects can be local outliers relative to this cluster, and 
#2) smaller than the maximum number of close by objects that can potentially be local outliers. In practice, such informations are generally not available, and taking n_neighbors=20 appears to work well in general.
set.seed(12345)
data(iris)
data<- iris
df<-iris[-5]

preProcValues <- preProcess(df, method = c("center", "scale"))
#dFTransformed <- predict(preProcValues, df)
dFTransformed <- df

seq(15,55,by=5)
lof<-lof(dFTransformed,c(seq(15,105,by=10)),cores=2)
lof.score <-as.data.frame(lof)
 
names(lof.score)<-make.names(names(lof.score))
data.score<-lof.score %>% gather(X15:X105, key = "k", value = "lof")
ggplot(data=data.score, aes(lof)) +  geom_density(aes(group=k, colour=k, fill=k),alpha = 0.3,linetype="solid") + ggtitle("LOF Score") 

lof.score$max <- apply(lof.score, 1, max)

 
#outlier threshold
thresh <- 0.95
Percentile = quantile(lof.score$max, probs = thresh)

data$error <- (lof.score$max>Percentile)


Lof.plot1 <- ggplot(data=data, aes(x=Sepal.Length, y=Sepal.Width,color=Species , shape=Species, size=(error)*0.1)) + geom_point()+  ggtitle("LOF Measures")
Lof.plot1

Lof.plot2 <- ggplot(data=data, aes(x=Petal.Length, y=Petal.Width,color=Species , shape=Species, size=(error)*0.1)) + geom_point()+  ggtitle("LOF Measures")
Lof.plot2


grid.arrange(cluster.plot1,SVM_One_Class.plot1,Auto_Encoder.plot1,Lof.plot1,cluster.plot2,SVM_One_Class.plot2,Auto_Encoder.plot2,Lof.plot2, ncol=4,nrow=2)
