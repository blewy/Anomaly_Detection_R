library(e1071)
library(caret)
library(NLP)
library(tm)
library(h2o)
library(data.table)
library(tidyverse)
library(ggplot2)
library(gridExtra)

# -------------------- data        ------------

rawfile<-read.csv("./data/train.csv",header=T) #Reading the csv file
im<-matrix((rawfile[2,2:ncol(rawfile)]), nrow=28, ncol=28) #For the 1st Image

im_numbers <- apply(im, 2, as.numeric)
image(1:28, 1:28, im_numbers, col=gray((0:255)/255))

# ------------  H2O autoencoders  -------------

cl <- h2o.init(
  max_mem_size = "4G",
  nthreads = 10)

data(iris)
df <- iris
dim(df)

label<-rawfile[,1]
df<-rawfile[,-1]

preProcValues <- preProcess(df, method = c("center", "scale"))
dfTransformed <- predict(preProcValues, df)

inTrain<-createDataPartition(1:nrow(dfTransformed),p=0.75,list=FALSE)

h2oData <- as.h2o(
  dfTransformed,
  destination_frame = "h2oData")

h2oData.train <- h2oData[inTrain,]
dim(h2oIris.train)
h2oData.test <- h2oData[-inTrain,]
xnames <- colnames(h2oData.train)
dim(h2oData.test)
folds <- createFolds(1:dim(h2oData.train)[1], k = 5)


## create parameters to try
hyperparams <- list(
  list(# M1
    hidden = c(50),
    input_dr = c(0),
    hidden_dr = c(0),
    activation_dr='RectifierWithDropout',
    l1_dr=0,
    l2_dr=0),
  list(# M2
    hidden = c(50),
    input_dr = c(.2),
    hidden_dr = c(.0),
    activation_dr='RectifierWithDropout',
    l1_dr=0,
    l2_dr=0),
  list(# M3
    hidden = c(100),
    input_dr = c(.2),
    hidden_dr = c(0),
    activation_dr='RectifierWithDropout',
    l1_dr=0,
    l2_dr=0),
  list(# M4
    hidden = c(100),
    input_dr = c(.2),
    hidden_dr = c(.5),
    activation_dr='RectifierWithDropout',
    l1_dr=0,
    l2_dr=0),
  list(# M5
    hidden = c(50,50),
    input_dr = c(.2),
    hidden_dr = c(.5,.5),
    activation_dr='RectifierWithDropout',
    l1_dr=0,
    l2_dr=0),
  list(# M6
    hidden = c(100,100),
    input_dr = c(.2),
    hidden_dr = c(.5,.5),
    activation_dr='RectifierWithDropout',
    l1_dr=0,
    l2_dr=0)
 
  )


fm <- lapply(hyperparams, function(v) {
  lapply(folds, function(i) {
    h2o.deeplearning(
      x = xnames,
      training_frame = h2oData.train[-i, ],
      validation_frame = h2oData.train[i, ],
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

length(Model)
length(MSE)

head(fm.res)

p.erate <- ggplot(fm.res, aes(reorder(Model,MSE,FUN=median), MSE)) +
  geom_boxplot() +
  stat_summary(fun.y = mean, geom = "point", colour = "red") +
  theme_classic()
p.erate

fm.res[, .(Mean_MSE = mean(MSE)), by = Model][order(Mean_MSE)]

best<-5

fm.final <- h2o.deeplearning(
  x = xnames,
  training_frame = h2oData.train,
  validation_frame = h2oData.test,
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
thresh <- 0.9
Percentile = quantile(error2a$Reconstruction.MSE, probs = thresh)

data$error <- (data$Reconstruction.MSE>Percentile)

Auto_Encoder.plot <- ggplot(data=data, aes(x=Sepal.Length, y=Sepal.Width,color=Species , shape=Species, size=(error)*0.1)) + geom_point()+  ggtitle("H2O Auto Encoder")
Auto_Encoder.plot

qnorm(0.95, mean = 0, sd = 1, lower.tail = TRUE, log.p = FALSE)

# ------------  svm 1-class -------------


# • you want that every datapoint is within (on the + side) or at the border of the hyperplane then set nu <= 1/n.

# • you want to go for the mode set nu=1 and all datapoints are outside (on the - side) of the hyperplane

# • you expect your sample to contain 10% outliers: nu = 1–0.1 = 0.9, hence, 90% of your datapoints will lie inside (on the + side) the hyperplane

label<-rawfile[,1]
df<-rawfile[,-1]


svm.model<-svm(df,y=NULL,
               type='one-classification',
               nu=0.05,
               scale=TRUE,
               kernel="radial")
#nu can be seen as outlier threshold
svm.model


svm.predtrain<-predict(svm.model,df)


data2<-cbind(label,svm.predtrain)

SVM_One_Class.plot<- ggplot(data=data, aes(x=Sepal.Length, y=Sepal.Width,color=Species , shape=Species, size=(1-svm.predtrain)*0.1)) + geom_point() +  ggtitle("SVM One-Class")


# remove species from the data to cluster
#iris2 <- iris[,1:4]
#


# ------------  K-means clustering -------------


# data(iris)
# df <- iris
# dim(df)

label<-rawfile[,1]
df<-rawfile[,-1]

range01 <- function(x, ...){
  (x - min(x, ...)) / (max(x, ...) - min(x, ...))
}

lapply(

nclust<-8

# 30 K Means Loop
se = seq(1,101,10)  
InerIC = rep(0, length(se))
for (k in 1:length(se)) 
{
  set.seed(42)
  groups = kmeans(df, se[k],iter.max = 1000, nstart = 50)
  InerIC[k] = groups$tot.withinss
  cat("\n Clusters : ",se[k],"\n")
} 

plot(InerIC, col ="blue", type ="b")
abline(v = nclust, col = "black", lty = 3)
text(8, 0.4, paste(c(nclust," Clusters"), collapse = " "),col = "black", adj = c(0,-0.1), cex = 0.7)


#Outra forma de avaliar o # de clusters poderá  com average silhouette, usualmente mais conservado

# Vamos calcular a silhoutte média de cada cluster para cada conjunto potencial de clusters.

nk = 2:15
set.seed(42)
SW = sapply(nk, function(k) {
  cluster.stats(dist(data), kmeans(data,
                                   centers=k)$cluster)$avg.silwidth
})
SW

# Visualizar o valor da **Silhouette** para cada número de clusters.
plot(nk, SW, type="b", xlab="Number of clusters", ylab="average silhouette width", col ="blue")
abline(v = nclust, col = "black", lty = 3)
text(nclust+0.5, 0.35, paste(c(nclust,"    Clusters")),col = "black", adj = c(0,-0.1), cex = 0.7)

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

cluster.plot <- ggplot(data=data3, aes(x=Sepal.Length, y=Sepal.Width,color=Species , shape=Species, size=(error_kmeans)*0.1)) + geom_point() +  ggtitle("Cluster Outliers")


grid.arrange(cluster.plot,SVM_One_Class.plot,Auto_Encoder.plot, ncol=2,nrow=2)

