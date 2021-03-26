
# Load the Dataset
churn <- read.csv("CustomerChurn.csv", stringsAsFactors = FALSE)

# Load libraries
library(caret)
library(rpart)
library(rpart.plot)
library(e1071)
library(naivebayes)
library(ggplot2) 
library(caret) 
library(cluster)
library(factoextra) 
library(fpc) 
library(Rtsne) 
library(pROC) 
library(ROCR)
library(tidyverse)
library(party)
library(plyr)
library(corrplot)
library(gridExtra)
library(MASS)
library(randomForest) 
library(class) 
library(party)
library(plyr)
library(corrplot)
library(ggplot2)
library(gridExtra)
library(caret)
library(MASS)
library(randomForest)

# Prepare the target variable

churn$Churn <- factor(churn$Churn)

# To get a general overview of our data, let's plot our target variable

plot(churn$Churn, main = "Churn")

# Let's also look at how our individual variables influence Churn
# We'll use stacked bar plots for our categorical variables and box-plots for our 
# numeric variables
# This code has been converted to comments to prevent clutter
###############################################################################
#                             Descriptive Analysis                            #
###############################################################################

#ggplot(data = churn, mapping = aes(x = gender,fill = Churn)) + geom_bar()
#ggplot(data = churn, mapping = aes(x = SeniorCitizen,fill = Churn)) + geom_bar()
#ggplot(data = churn, mapping = aes(x = Partner,fill = Churn)) + geom_bar()
#ggplot(data = churn, mapping = aes(x = Dependents,fill = Churn)) + geom_bar()
#ggplot(data = churn, mapping = aes(x = Churn, y = tenure)) + geom_boxplot()
#ggplot(data = churn, mapping = aes(x = PhoneService,fill = Churn)) + geom_bar()
#ggplot(data = churn, mapping = aes(x = MultipleLines,fill = Churn)) + geom_bar()
#ggplot(data = churn, mapping = aes(x = InternetService,fill = Churn)) + geom_bar()
#ggplot(data = churn, mapping = aes(x = OnlineSecurity,fill = Churn)) + geom_bar()
#ggplot(data = churn, mapping = aes(x = OnlineBackup,fill = Churn)) + geom_bar()
#ggplot(data = churn, mapping = aes(x = DeviceProtection,fill = Churn)) + geom_bar()
#ggplot(data = churn, mapping = aes(x = TechSupport,fill = Churn)) + geom_bar()
#ggplot(data = churn, mapping = aes(x = StreamingTV,fill = Churn)) + geom_bar()
#ggplot(data = churn, mapping = aes(x = StreamingMovies,fill = Churn)) + geom_bar()
#ggplot(data = churn, mapping = aes(x = Contract,fill = Churn)) + geom_bar()
#ggplot(data = churn, mapping = aes(x = PaperlessBilling,fill = Churn)) + geom_bar()
#ggplot(data = churn, mapping = aes(x = PaymentMethod,fill = Churn)) + geom_bar()
#ggplot(data = churn, mapping = aes(x = Churn, y = MonthlyCharges)) + geom_boxplot()
#ggplot(data = churn, mapping = aes(x = Churn, y = TotalCharges)) + geom_boxplot()

# We will set up convenience vectors for our numeric and categorical variables]

cats <- c("gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", 
          "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
          "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", 
          "Contract", "PaperlessBilling", "PaymentMethod")
churn[,cats] <- lapply(X = churn[,cats], FUN = factor)

nums <- c("tenure", "MonthlyCharges", "TotalCharges")
churn[,nums] <- lapply(X = churn[,nums], FUN = as.numeric)

# We will combine cats and nums to create vars, which we will use to predict churn

vars <- c(cats, nums)

# We will now check to see if we have any missing values

any(is.na(churn))

# There are indeed missing values, so we must remove them

churn <- na.omit(churn)

###########################################################
#                       Clustering Analysis               #
###########################################################


# Unsupervised learning using Cluster Analysis



# Loading the RData file that contains the necessary functions
# which will be used during validation of HCA and KMeans

#load("Clustering.RData")

#Reading the given data into a dataset named 'cc'

cc<- read.csv(file = "CustomerChurn.csv")

# viewing the structure of the data
str(cc)

# viewing the summary of the data
summary(cc)


# creating a vector named 'facs1' containing categorical variables 
facs1<- c("customerID","gender", "SeniorCitizen","Partner","Dependents","PhoneService",
          "MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection",
          "TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod",
          "Churn")


# creating a vector named 'nums1' containing numerical variables

nums1 <- names(cc)[!names(cc) %in% facs1]

#or we can use the below code 
#nums1<-c("tenure","MonthlyCharges","TotalCharges")

# viewing what is in nums and facs 
facs1
nums1

# converting the categorical variables to factors
cc[ ,facs1] <- lapply(X = cc[ ,facs1], FUN = factor)

# Handling the missing values

any(is.na(cc)) #see if there are any missing values in the data.

cc_nn <- na.omit(cc)#omitting the missing values

any(is.na(cc_nn)) # confirming if the missing values are removed.

#Standardizing the data 

cen_cc <- preProcess(x = cc_nn,
                     method = c("center", "scale"))
cc_nn <- predict(object = cen_cc,
                 newdata = cc_nn)

#Visualizing the target variable 'Churn'

plot(cc_nn$Churn)

#Dimensionality reduction

pca <- prcomp(x = cc_nn[, nums1],
              scale. = TRUE)

pca

screeplot(pca, type = "lines")

# We do not require any dimensionality reductions since we have only 3 numeric variables

#Identifying outliers

cc_box <- boxplot(x = cc_nn[ ,nums1], 
                  main = "Numerical")

cc_box$out # outlier values

#There are no outliers in the data.

#Finding the correlation between the numerical variables

cor(x=cc_nn[ , nums1])

#strong correlation between MonthlyCharges and TotalCharges


#Hierarchical Clustering--------------------------------------

#we transform the data to normal distribution

cen_yeojo <- preProcess(x= cc_nn, method ="YeoJohnson")
cc_yeojo <- predict(object = cen_yeojo,
                    newdata = cc_nn)

#verifying if the variables are normalized (considering TotalCharges)
hist(cc_nn$TotalCharges) #Before normalization
hist(cc_yeojo$TotalCharges) #After normalization

#removing variables that are not of interest (customerID) and the target variable (Churn)
mydata<-cc_yeojo[ ,c(-1,-21)] #removing variables customerID and Churn 

names(mydata)

mydata2<-mydata[ ,-19] # removing totalCharges since it has high correlation with 
#MonthlyCharges.

names(mydata2)

facs2<- c("gender", "SeniorCitizen","Partner","Dependents","PhoneService",
          "MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection",
          "TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod"
)
nums2<-c("tenure","MonthlyCharges")



# Using gower distance method to calculate distances and similarity since there are 
#mixed variables

hdist2 <- daisy(x = mydata2, metric= "gower")

summary(hdist2)

#Agglomerative clustering

#Measuring dissimilarity between clusters using single, complete, 
# average, centroid and wards linkage methods

#Single Linkage

sing1<- hclust(hdist2,method = "single")

#dendrogram for single linkage

plot(sing1, sub = NA, xlab = NA, 
     main = "Single Linkage")

#Identifying cluster boxes for k= 5 clusters
rect.hclust(tree = sing1, # hclust object
            k = 5, # # of clusters
            border = hcl.colors(5)) # k colors for boxes

# Single linkage dendrogram is long  and loosely packed clusters.
#It seems that it does not produce a good clustering solution.

#Vector of single Clusters
single1_clusters <- cutree(tree = sing1, k = 5)

#Complete Linkage
comp1<-hclust(hdist2, method ="complete")

#Plotting the dendrogram for complete linkage

plot(comp1, sub = NA, xlab = NA, 
     main = "complete Linkage")

#Identifying clusters and dividing them into boxes of k=5 clusters
rect.hclust(tree = comp1, k = 5, 
            border = hcl.colors(5))

#Creating vector of complete clusters
complete1_clusters <- cutree(tree = comp1, k = 5)

#average linkage

avg1<- hclust(d = hdist2, 
              method = "average")

#Plotting the dendrogram for average linkage
plot(avg1, 
     sub = NA, xlab = NA, 
     main = "Average Linkage")

#Identifying clusters and dividing them into boxes of k=5 clusters
rect.hclust(tree = avg1, k = 5, 
            border = hcl.colors(5))

#Creating vector of average clusters
avg1_clusters <- cutree(tree = avg1, k = 5)


#centroid Linkage


cent1<- hclust(d = hdist2 ^ 2, 
               method = "centroid")

#Plotting the dendrogram for centroid linkage
plot(cent1, 
     sub = NA, xlab = NA, 
     main = "Centroid Linkage")

#Identifying clusters and dividing them into boxes of k=5 clusters

rect.hclust(tree = cent1, k = 5, 
            border = hcl.colors(5))

# creating a vector for centroid clusters 
cent1_clusters <- cutree(tree = cent1, k = 5)

#wards method

wards1 <- hclust(d = hdist2, 
                 method = "ward.D2")

#Plotting the dendrogram for wards linkage
plot(wards1, 
     xlab = NA, sub = NA, 
     main = "Ward's Method")

#Identifying clusters and dividing them into boxes of k=5 clusters
rect.hclust(tree = wards1, k = 5, 
            border = hcl.colors(5))

# creating a vector for wards clusters 
wards1_clusters <- cutree(tree = wards1, k = 5)

#Visualizing the solutions of the above methods

#Reducing the dimensionality using Rtsne function from Rtsne package

ld_dist <- Rtsne(X = hdist2, 
                 is_distance = TRUE)

lddf_dist <- data.frame(ld_dist$Y)

#Plotting the reduced dimension 

#Single linkage

ggplot(data = lddf_dist, 
       mapping = aes(x = X1, y = X2)) +
  geom_point(aes(color = factor(single1_clusters))) +
  labs(color = "Cluster")

# Complete Linkage
ggplot(data = lddf_dist, 
       mapping = aes(x = X1, y = X2)) +
  geom_point(aes(color = factor(complete1_clusters))) +
  labs(color = "Cluster")

#centroid linkage

ggplot(data = lddf_dist, 
       mapping = aes(x = X1, y = X2)) +
  geom_point(aes(color = factor(cent1_clusters))) +
  labs(color = "Cluster")

#avg_Cluster
ggplot(data = lddf_dist, 
       mapping = aes(x = X1, y = X2)) +
  geom_point(aes(color = factor(avg1_clusters))) +
  labs(color = "Cluster")

# Ward's Method
ggplot(data = lddf_dist, 
       mapping = aes(x = X1, y = X2)) +
  geom_point(aes(color = factor(wards1_clusters))) +
  labs(color = "Cluster")


#Describing cluster solutions for wards, complete and average methods

aggregate(x = mydata2[ ,nums2], 
          by = list(wards1_clusters),
          FUN = mean)

aggregate(x = mydata2[ ,nums2], 
          by = list(complete1_clusters),
          FUN = mean)

aggregate(x = mydata2[ ,nums2], 
          by = list(avg1_clusters),
          FUN = mean)

aggregate(x = mydata2[ ,facs2], 
          by = list(single1_clusters), 
          FUN = table)


#ac <- function(x) {
# agnes(mydata2, method = x)$ac
#}

#using agnes function from the cluster package

#w<-agnes(mydata,method = "ward")
#c<-agnes(mydata,method = "complete")
#a<-agnes(mydata,method = "average")
#s<-agnes(mydata,method = "single")

#w$ac
#c$ac
#a$ac
#s$ac

##Cluster validation

#We are comparing the distribution of clusters with respect to churn 
#in complete, wards and average clustering methods.

#k= 5

table(CustomerChurn = cc_nn$Churn, 
      Clusters = complete1_clusters)

table(CustomerChurn = cc_nn$Churn, 
      Clusters = wards1_clusters)

table(CustomerChurn = cc_nn$Churn, 
      Clusters = avg1_clusters)

#Customer churn is more for the observations in cluster 3

table(CustomerChurn = cc_nn$Churn, 
      Clusters = single1_clusters)

#Adjusted Rand Index

cluster.stats(d = hdist2, # distance matrix
              clustering = wards1_clusters, # cluster assignments
              alt.clustering = as.numeric(cc_nn$Churn))$corrected.rand # known groupings

cluster.stats(d = hdist2, # distance matrix
              clustering = complete1_clusters, # cluster assignments
              alt.clustering = as.numeric(cc_nn$Churn))$corrected.rand # known groupings

cluster.stats(d = hdist2, # distance matrix
              clustering = avg1_clusters, # cluster assignments
              alt.clustering = as.numeric(cc_nn$Churn))$corrected.rand # known groupings

cluster.stats(d = hdist2, # distance matrix
              clustering = single1_clusters, # cluster assignments
              alt.clustering = as.numeric(cc_nn$Churn))$corrected.rand #known groupings

# Though the Rand index value is high for wards linkage method,
#we are not using Rand Index measure to validate the clusters as it 
#is not close to 1 or -1.

#Internal Validation

#Cophenetic correlation using the original distance and the cophenetic distance
#of each of the clustering methods

# Single Linkage
cor(x = hdist2, y = cophenetic(x = sing1))

# Complete Linkage
cor(x = hdist2, y = cophenetic(x = comp1))

# Average Linkage
cor(x = hdist2, y = cophenetic(x = avg1))

# Centroid Linkage
cor(x = hdist2 ^ 2, y = cophenetic(x = cent1))

# Ward's Method
cor(x = hdist2, y = cophenetic(x = wards1))

# Cophenetic correlation is high for average linkage method

#Looking for the elbow point based on 
#the within cluster some of squares(Wss) plot


# Bringing the wss plot and sil plot functions from the clustering.RData file
load("Clustering.RData") 

wss_plot(dist_mat = hdist2, # distance matrix
         method = "hc", # HCA
         hc.type = "average", # linkage method
         max.k = 15) # maximum k value
## Strict Elbow at k =  5

wss_plot(dist_mat = hdist2, # distance matrix
         method = "hc", # HCA
         hc.type = "ward.D2", # linkage method
         max.k = 15) # maximum k value

# There is no strict elbow point for wards linkage in order to
#determine the number of Clusters
# so, we would like to take the average linkage method.

#wss_plot(dist_mat = hdist2, # distance matrix
#         method = "hc", # HCA
#         hc.type = "complete", # linkage method
#         max.k = 15) # maximum k value
#Strict elbow at k=4

## Silhouette Method

# Plotting average silhouette width for different k values using sil_plot function
# and finding the maximum average silhouette width for a k

# Hierarchical Cluster Analysis (method = "hc")
sil_plot(dist_mat = hdist2, # distance matrix
         method = "hc", # HCA
         hc.type = "ward.D2", # average linkage
         max.k = 15) # maximum k value

#Silhouette width is maximum at K=2

sil_plot(dist_mat = hdist2, # distance matrix
         method = "hc", # HCA
         hc.type = "average", # average linkage
         max.k = 15) # maximum k value

#Silhouette width is maximum at K=2
#-------------------------------------------------------------------------------

#K-means clustering analysis

#Standardizing yeo johnson transformed data 

cen_cc <- preProcess(x = cc_yeojo,
                     method = c("center", "scale"))
cc_yjcc <- predict(object = cen_cc,
                   newdata = cc_yeojo)

cc_yjcc

any(is.na(cc_yjcc)) # no missing values

#Initializing random seed using set.seed() for the initial cluster centers

set.seed(29619122)

#Kmeans function

#Our initial value for number of clusters is 5

kmeans1 <- kmeans(x = cc_yjcc[,nums2], # data
                  centers = 5, # # of clusters
                  trace = FALSE, 
                  nstart = 30)

kmeans1 #Summary of kmeans


barplot(kmeans1$size, col="Steel blue")

kmeans1$size # frequency distribution of the clusters 

#visualizing the cluster solution using fviz_cluster function

fviz_cluster(object = kmeans1, 
             data = cc_yjcc[ ,nums2])


#autoplot(kmeans1,mydata,frame=TRUE)

## Describe the Cluster Solution

clus_means_kMC1 <- aggregate(x = cc_yjcc[ ,nums2], 
                             by = list(kmeans1$cluster), 
                             FUN = mean)
clus_means_kMC1

# plotting the scaled cluster centers using matplot function 

#-------------------------------------------------------------------------------

# Cluster validation

#Distribution of clusters with respect to churn 
#in KMeans analysis.

table(Customerchurn = cc_nn$Churn, 
      Clusters = kmeans1$cluster)

barplot(table(Customerchurn = cc_nn$Churn, 
              Clusters = kmeans1$cluster))

#External validation using Adjusted Rand

# k-Means Clustering (kMC, k = 5)
cluster.stats(d = dist(cc_yjcc[ ,nums2]), # distance matrix for data used
              clustering = kmeans1$cluster, # cluster assignments
              alt.clustering = as.numeric(cc_nn$Churn))$corrected.rand

#internal validation 

# wss plot
set.seed(29619122)

#Looking for the elbow point based on 
#the within cluster some of squares(Wss) plot for KMeans analysis

wss_plot(scaled_data = cc_yjcc[ ,nums2], # dataframe
         method = "kmeans", # kMC
         max.k = 15, # maximum k value
         seed_no = 29619122) 

#Elbow at k=4

#Silplot

# Plotting average silhouette width for different k values using sil_plot function
# and finding the maximum average silhouette width for a k

sil_plot(scaled_data = cc_yjcc[ ,nums2], # scaled data
         method = "kmeans", # kMC
         max.k = 15, # maximum k value
         seed_no = 29619122) # seed value for set.seed()
# Maximum Average Silhouette Width is at k=4

#other validation techniques

c_stats1<- c("max.diameter", "min.separation", 
             "average.between", "average.within",
             "dunn")

# Hierarchical Clustering
# Based on Coph. Cor: Average Linkage
# Based on WSS:  5,    Sil: 2

# Obtain Cluster Statistics
stats_HCA1 <- cluster.stats(d = hdist2, 
                            clustering = avg1_clusters)


## k-Means Clustering (kMC)
# kMC, WSS: 4;       Sil: 4
set.seed(29619122)

#kMeans with k=4
kmeans4 <- kmeans(x = cc_yjcc[ ,nums2], 
                  centers = 4,
                  nstart = 30,
                  trace = FALSE)

# Obtain Cluster Statistics
stats_kMC1 <- cluster.stats(d = dist(cc_yjcc[,nums2]), 
                            clustering = kmeans4$cluster)

#dunn index
cbind(HCA = stats_HCA1, 
      kMC = stats_kMC1)["dunn",]

#Dunn index is high for HCA. In this content HCA is better.

cbind(HCA = stats_HCA1, 
      kMC = stats_kMC1)["max.diameter",]

#Diameter of the clusters is small for HCA. In this context HCA is better.

cbind(HCA = stats_HCA1, 
      kMC = stats_kMC1)["min.separation",]

#Large separation between the clusters of HCA. In this context HCA is better.

cbind(HCA = stats_HCA1, 
      kMC = stats_kMC1)["average.between",]

#Large average distance between clusters is for KMeans. In this context KMeans is better.

cbind(HCA = stats_HCA1, 
      kMC = stats_kMC1)["average.within",]

#small average distance within clusters is found for HCA. In this context HCA is better.

# Overall, based on all the results of validation, HCA does a better job in clustering.

###############################################################################
#                             Naive Bayes                                     #
###############################################################################
# Initialize a seed to be used in calculations

set.seed(29619122)

# With the seed initialized, we can now create our training indices
# We will use a p of 0.85 to get a high amount of training data for our model

sub <- createDataPartition(y = churn$Churn, p = 0.85, list = FALSE)

# We will create 2 dataframes - one training and one testing

train <- churn[sub, ]
test <- churn[-sub, ]

cats <- c("gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", 
          "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
          "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", 
          "Contract", "PaperlessBilling", "PaymentMethod")
churn[,cats] <- lapply(X = churn[,cats], FUN = factor)

nums <- c("tenure", "MonthlyCharges", "TotalCharges")
churn[,nums] <- lapply(X = churn[,nums], FUN = as.numeric)
# With the initial setup complete, we can now move into Naive Bayes Classification

# To begin this process, we must first check correlation

cor(x = churn[ ,nums])

# Total charges correlates strongly with both tenure and monthly charges, so we'll
# want to remove it.

vars <- vars[!vars %in% "TotalCharges"]
vars

# We will now want to check our numeric variables to see if they are normally distributed

hist(x = churn$MonthlyCharges)
hist(x = churn$tenure)

# Neither is normally distributed, so we'll need to transform them
# Tenure has 0-values, so we'll want to use a Yeo-Johnson transformation.

Norm <- preProcess(x = churn[ ,vars], method = c("YeoJohnson", "center", "scale"))

# Training and Testing datasets have already been made, so we'll apply 
# transformations directly to them

train_NB <- predict(object = Norm, newdata = train)

test_NB <- predict(object = Norm, newdata = test)

# We must now determine if Laplace smoothing needs to be applied

aggregate(train_NB[ ,cats], by = list(train_NB$Churn), FUN = table)

# There are no 0-categories, so Laplace smoothing does not need to be applied!

# Next, we will ceate our Naive Bayes model

NB_model <- naiveBayes(x = train_NB[ ,vars], y = train_NB$Churn, laplace = 0)
NB_model

# Using our model, we will now generate class predictions

NB.train <- predict(object = NB_model, newdata = train_NB[ ,vars], type = "class")
head(NB.train)

# We will now use a confusion matrix to generate performance measures

NB_train_conf <- confusionMatrix(data = NB.train, reference = train_NB$Churn, 
                                 positive = "Yes", mode = "everything")
NB_train_conf
NB_train_conf$byoverall
# We will now go through the same process for our testing model

NB.test <- predict(object = NB_model, newdata = test_NB[ ,vars], type = "class")

NB_test_conf <- confusionMatrix(data = NB.test, reference = test_NB$Churn,
                                positive = "Yes", mode = "everything")

NB_test_conf

# We'll now test our overall performance based on accuracy and kappa values
# This will show the general accuracy of our model and our model's accuracy 
# accounting for chance

NB_test_conf$overall[c("Accuracy", "Kappa")]

# We'll also observe performance by class

NB_test_conf$byClass

# Let's now test goodness of fit for both overall and class-level performance

# Overall
cbind(Training = NB_train_conf$overall, Testing = NB_test_conf$overall)

# By class
cbind(Training = NB_train_conf$byClass, Testing = NB_test_conf$byClass)

# Our results show that the model is balanced

# Next, we'll look at variable importance

# First, we'll set up the grid to meet Naive Bayes' needs
# NOTE: Some code for the variable importance was obtained from the StackOverlow
# page: "Determine Variables of Importance in Naive Bayes Model".
# Code from this website will be preceded with a comment 'SOF'

# SOF
grids <- data.frame(usekernel=TRUE,laplace = 0,adjust=1)

grids

# Next, we'll set up our ctrl object

ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3, search = "grid")

set.seed(29619122)

# With this set up, we can make our NB_Fit object using the grids and ctrl objects
# Our method will be naive_bayes
# SOF

NB_Fit <- train(form = Churn ~ .,
                data = train[ ,c(vars, "Churn")],
                method = "naive_bayes",
                trControl = ctrl,
                tuneGrid = grids)

# We can now check and plot our variable importance

NB_Imp <- varImp(NB_Fit)
plot(NB_Imp)

###########################################################
#                       Random Forest                     #
###########################################################
# Load the data
churn_RF <- read.csv('CustomerChurn.csv')

# view the structure
str(churn_RF)

# Check for missing values
sapply(churn_RF, function(x) sum(is.na(x)))
churn_RF <- churn_RF[complete.cases(churn_RF), ]

# remove non useful columns
churn_RF$customerID <- NULL
churn_RF$TotalCharges<- NULL


# Partition the data set into training and testing data set
Train_RF<- createDataPartition(churn_RF$Churn,p=0.85,list=FALSE)
set.seed(29619122)
training_RF<- churn_RF[Train_RF,]
testing_RF<- churn_RF[-Train_RF,]


# build the random forest model
Random_forest <- randomForest(Churn ~., data = training_RF)
print(Random_forest)
?randomForest

pred_Randomforest1 <- predict(Random_forest, testing_RF)
caret::confusionMatrix(pred_Randomforest1, testing_RF$Churn)

# tune the model
tune_model <- tuneRF(training_RF[, -18], training_RF[, 18], stepFactor = 0.5, plot = TRUE, ntreeTry = 200, trace = TRUE, improve = 0.05)


Random_forest2 <- randomForest(Churn ~., data = training_RF, ntree = 199, mtry = 3, importance = TRUE, proximity = TRUE)
print(Random_forest2)


pred_Randomforest2 <- predict(Random_forest2, testing_RF)
caret::confusionMatrix(pred_Randomforest2, testing_RF$Churn)


# Importantce Feature Plot

varImpPlot(Random_forest2, sort=T, n.var = 18, main = 'Feature Importance plot')

###########################################################
#                           KNN                           #
###########################################################

## Classification Analysis
## k-Nearest Neighbors (kNN)
#------------------------------------------
#------------------------------------------


## Load Data
cc2 <- read.csv(file = "CustomerChurn.csv",
                stringsAsFactors = TRUE)

#Remove CustomerID variable, it is not needed

cc3 <- cc2[,-1]

## Data Exploration & Preparation
str(cc3)

## Prepare Target (Y) Variable
# We can convert our class variable,
# Churn to a nominal factor variable
# Note: No = 1, Yes = 2
cc3$Churn <- factor(x = cc3$Churn,
                    levels = c("No", "Yes"))

# plot the distribution of the target variable
plot(cc3$Churn, main = "Churn3")

## Prepare Predictor (X) Variables
names(cc3)[sapply(X = cc3, 
                  FUN = is.integer)]

lapply(X = cc3[,names(cc3)[sapply(X = cc3, 
                                  FUN = is.integer)]],
       FUN = table)

# We have two categorical variables, SeniorCitizen and
# tenure. We can keep SeniorCitizen as-is, since it is already binary. 
#Also we will keep tenure as-is, since it is an integer but it clearly takes
# so many values and will not be considered as factor variables

# cc3$tenure <- factor(x = cc3$tenure)

#set up a vector of the variable names
nums <- names(cc3)[names(cc3) %in% c("MonthlyCharges", "TotalCharges","tenure")]



## Data Preprocessing & Transformation

## 1. Remove missing values
any(is.na(cc3))
summary(cc3)
#there are any na values in "TotalCharges". Remove them
cc4 <- na.omit(cc3)
summary(cc4)

## 2. Redundant Variables


# obtain the correlation matrix for our numeric predictor variables
cor_vars <- cor(x = cc4[ ,nums])

#corelation is checked for only numeric values

symnum(x = cor_vars,
       corr = TRUE)

high_corrs <- findCorrelation(x = cor_vars, 
                              cutoff = .75, 
                              names = TRUE)
high_corrs

#There is no redundant varibale

#Below step is not required beacuse there are no redundatan variables.
#nums <- nums[!nums %in% high_corrs]
#Since there are no redundant variables, nums vector will remain as is

## 3. Rescale Numeric Variables
# kNN has been shown to perform well with min-max (range) normalization, converting
# the numeric variables to range between 0 and 1. We can use the preProcess()
# and predict() functions and save the rescaled data as a new dataframe, 
# cc4_mm.

cen4_mm <- preProcess(x = cc4[ ,nums],
                      method = "range")
cc4_mm <- predict(object = cen4_mm,
                  newdata = cc4)

#for min max even the binary variables can be taken into account, it will
#give the same result because min max will try and put the values in 0-1 range

## 4. Binarization

# We will binarize the factor variable:
#Below: Check the levels for individual class variables
#nlevels(cc4$gender) # class levels
#nlevels(cc4$Partner)
#nlevels(cc4$Dependents)
#nlevels(cc4$PhoneService)
#nlevels(cc4$MultipleLines)
#nlevels(cc4$InternetService)
#nlevels(cc4$OnlineSecurity)
#nlevels(cc4$OnlineBackup)
#nlevels(cc4$DeviceProtection)
#nlevels(cc4$TechSupport)
#nlevels(cc4$StreamingTV)
#nlevels(cc4$StreamingMovies)
#nlevels(cc4$Contract)
#nlevels(cc4$PaperlessBilling)
#nlevels(cc4$PaymentMethod)

#Creating dummy variables for individual class variables
#cats1 <- dummyVars(formula =  ~ gender,
#                 data = cc4)
#cats1_dums <- predict(object = cats1, 
#                    newdata = cc4)

#cats2 <- dummyVars(formula =  ~ Partner,
#                  data = cc4)
#cats2_dums <- predict(object = cats2, 
#                     newdata = cc4)
#cats3 <- dummyVars(formula =  ~ Dependents,
#                  data = cc4)
#cats3_dums <- predict(object = cats3, 
#                     newdata = cc4)
#cats4 <- dummyVars(formula =  ~ PhoneService,
#                  data = cc4)
#cats4_dums <- predict(object = cats4, 
#                     newdata = cc4)
#cats5 <- dummyVars(formula =  ~ MultipleLines,
#                  data = cc4)
#cats5_dums <- predict(object = cats5, 
#                     newdata = cc4)

#Convert all the class variables into dummy variables
cats101 <- dummyVars(formula =  "~ .",
                     data = cc4)
cats101_dums <- predict(object = cats101, 
                        newdata = cc4)


# Combine binarized variables with data (transformed numeric variables, factor 
# target variable)

cc4_mm_dum <- data.frame(cc4_mm,cats101_dums)
names(cc4_mm_dum)

cc4_mm_dum_final <- cc4_mm_dum[,-c(1:19,65:67)]

# Create vars vector of the names of the variables to use as input to the kNN model
vars101 <- names(cc4_mm_dum_final)[!names(cc4_mm_dum_final) %in% "Churn"]
vars101

## 5 Training & Testing

set.seed(29619122) # initialize the random seed

# Create partition train:test ratio should be 85%:15%
sub101 <- createDataPartition(y = cc4_mm_dum_final$Churn, 
                              p = 0.85, 
                              list = FALSE)

# Subset the rows of the cc4_mm_dum dataframe to include the row numbers in the
#sub object to create the train dataframe
train101 <- cc4_mm_dum_final[sub101, ] 
train102 <- train101[!names(train101) %in% "Churn"]
train102

test101 <- cc4_mm_dum_final[-sub101, ]
test102 <- test101[!names(test101) %in% "Churn"]
test102

## 6. Analysis


## First, we can try using a 'best guess' value of k (square root of the number 
# of training observations)
ceiling(sqrt(nrow(train101)))
# 78

## Naive Model Building : Since K is even, take the odd number to make classification better
#Hence take k=77
knn.pred101 <- knn(train102,test102, 
                   cl = train101$Churn, 
                   k = 77)

conf_basic <- confusionMatrix(data = knn.pred101, # vector of Y predictions
                              reference = test101$Churn, # actual Y
                              positive = "Yes", # positive class for class-level performance
                              mode = "everything") # all available measures

conf_basic
accuracy <- mean(observed.classes == predicted.clases)

### Hyperparameter Tuning 

# Note: specifying tuneLength = 15 and no particular hyperparameter search method will 
# perform a default grid search

# By default, the train() function will determine the 'best' model based on Accuracy 
# for classification and RMSE for regression. For classification models, the Accuracy and Kappa 
# are automatically computed and provided. 

# Set up a trainControl object (named ctrl) using the trainControl() 
# function in the caret package. We specify that we want to perform 10-fold cross 
# validation, repeated 3 times. We use this object as input to the trControl argument
# in the train() function below.
ctrl <- trainControl(method = "repeatedcv",
                     number = 10,
                     repeats = 3)

# Next, we initialize a random seed for 
# our cross validation
set.seed(29619122)

# Then, we use the train() function to train the kNN model using 10-Fold Cross 
# Validation (repeated 3 times). We set tuneLength = 15 to try the first 15
# default values of k (odd values from k = 5:33)
knnFit101 <- train(x = train102,
                   y = train101$Churn, 
                   method = "knn", 
                   trControl = ctrl, 
                   tuneLength = 50)

knnFit101

# Plot the train() object (knnFit1) using the plot() function
# to view a plot of the hyperparameter,k, on the x-axis and the Accuracy 
# on the y-axis.
plot(knnFit101)

# We can view the confusion matrix showing the average performance of the model
# across resamples
confusionMatrix(knnFit101)

### Model Performance

# Finally,use our best tuned model to predict the testing data.First, we use the predict() function to 
# predict the value of the median_val variable 
# using the model we created using the train()function, knnFit1 model and the true 
# classes of the median_val in the test dataframe.
outpreds <- predict(object = knnFit101, 
                    newdata = test102)

#  set mode ="everything" to obtain all available performance measures
conf_tuned <- confusionMatrix(data = outpreds, 
                              reference = test101$Churn, 
                              positive = "Yes",
                              mode = "everything")
conf_tuned


# We can describe the overall performance 
# based on our accuracy and kappa values.

conf_tuned$overall[c("Accuracy", "Kappa")]

# We can describe class-level performance for the different class levels. Note,
# above, we set positive = "Yes", since we are more interested in predicting above median
# properties than below median
conf_tuned$byClass

## Comparing Base & Tuned Models

# Overall Model Performance
cbind(Base = conf_basic$overall,
      Tuned = conf_tuned$overall)

# Class-Level Model Performance
cbind(Base = conf_basic$byClass,
      Tuned = conf_tuned$byClass)

mean(outpreds == test101$Churn)

plot(knnFit101, print.thres = 0.5, type="S")

outpreds <- predict(object = knnFit101, 
                    newdata = test102, type = "prob")


#conf_basic
#fp = 55
#fn = 151
#tp = 129
#tn = 719
#fpr = fp / (fp + tn)
#tpr = tp / (tp + fn)
#AUC <- 1/2 - fpr/2 + tpr/2
#AUC
#conf_basic$table
#conf_tuned$table

#summary(cc4$churn)

cc4_mm_dum_final_churn <- table(cc4_mm_dum_final$Churn)
cc4_mm_dum_final_churn

#Performed knn with k where accuracy is best i.e. at 19
#But the accuracy and kappa are better with k = 77

#knn.pred201 <- knn(train102,test102, 
#                   cl = train101$Churn, 
#                   k = 19)

#conf_basic201 <- confusionMatrix(data = knn.pred201, # vector of Y predictions
#                                 reference = test101$Churn, # actual Y
 #                                positive = "Yes", # positive class for class-level performance
  #                               mode = "everything")
#conf_basic201

#knnFit201 <- train(x = train102,
 #                  y = train101$Churn, 
  #                 method = "knn", 
   #                trControl = ctrl, 
    #               tuneLength = 50)

#plot(knnFit201)
#confusionMatrix(knnFit201)

### Hyperparameter Tuning 

#ctrl <- trainControl(method = "repeatedcv",
 #                    number = 10,
  #                   repeats = 3)
#set.seed(29619122)

### Model Performance

#outpreds1 <- predict(object = knnFit201, 
 #                   newdata = test102)

#  set mode ="everything" to obtain all available performance measures
#conf_tuned1 <- confusionMatrix(data = outpreds1, 
 #                             reference = test101$Churn, 
  #                            positive = "Yes",
   #                           mode = "everything")
#conf_tuned1

#conf_tuned1$overall[c("Accuracy", "Kappa")]
#conf_tuned1$byClass
## Comparing Base & Tuned Models

# Overall Model Performance
#cbind(Base = conf_basic201$overall,
 #     Tuned = conf_tuned1$overall)

# Class-Level Model Performance
#cbind(Base = conf_basic201$byClass,
 #     Tuned = conf_tuned1$byClass)

#mean(outpreds1 == test101$Churn)

#plot(knnFit201, print.thres = 0.5, type="S")

#outpreds2 <- predict(object = knnFit201, 
 #                   newdata = test102, type = "prob")



############################## End  ###########################################


