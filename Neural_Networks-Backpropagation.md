Neural Networks - Backpropagation
================

Introduction
------------

In a previous exercise, I made a neural work to predict animal types using neuro-evolution to optimize the network. This time I will do the same, but instead of neuro-evolution, I will use backpropagation and gradient descent to train the network.

I found the free online ["book"](http://neuralnetworksanddeeplearning.com/) by Michael A. Nielsen and [the neural network series](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) by 3BLUE1BROWN very helpful to understand how gradient descent and backpropagation work.

The structure
-------------

-   Data and Data Preparations
-   Creating the necessary functions
    -   Neural network
    -   Backpropagation
    -   Stochastic gradient descent
    -   Update mini batch
-   Train the model.
-   Evaluate the model.

Data and Data Preperation
-------------------------

### Select and Examine the Data

Importing the ["Zoo"](https://archive.ics.uci.edu/ml/datasets/Zoo) data set containing information about different animals. The data set consists of 18 variables, the animal name, and the type and 16 features. All features are binary, except for the "legs" variable which is nominal with 6 categories. There are 7 animal types/classes and after some training, the classifier should be able to sort the animals into the right class based on the features.

``` r
set.seed(1)
library(readr)
```

    ## Warning: package 'readr' was built under R version 3.5.2

``` r
library(compiler)

#import dataset
zoo <- read.csv("zoo.txt", header=T)

# have a look at the data structure
str(zoo)
```

    ## 'data.frame':    101 obs. of  18 variables:
    ##  $ animal_name: Factor w/ 100 levels "aardvark","antelope",..: 1 2 3 4 5 6 7 8 9 10 ...
    ##  $ hair       : int  1 1 0 1 1 1 1 0 0 1 ...
    ##  $ feathers   : int  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ eggs       : int  0 0 1 0 0 0 0 1 1 0 ...
    ##  $ milk       : int  1 1 0 1 1 1 1 0 0 1 ...
    ##  $ airborne   : int  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ aquatic    : int  0 0 1 0 0 0 0 1 1 0 ...
    ##  $ preditor   : int  1 0 1 1 1 0 0 0 1 0 ...
    ##  $ toothed    : int  1 1 1 1 1 1 1 1 1 1 ...
    ##  $ backboned  : int  1 1 1 1 1 1 1 1 1 1 ...
    ##  $ breathes   : int  1 1 0 1 1 1 1 0 0 1 ...
    ##  $ venomous   : int  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ fins       : int  0 0 1 0 0 0 0 1 1 0 ...
    ##  $ legs       : int  4 4 0 4 4 4 4 0 0 4 ...
    ##  $ tails      : int  0 1 1 0 1 1 1 1 1 0 ...
    ##  $ domestic   : int  0 0 0 0 0 0 1 1 0 1 ...
    ##  $ catsize    : int  1 1 0 1 1 1 1 0 0 0 ...
    ##  $ type       : int  1 1 4 1 1 1 1 4 4 1 ...

``` r
head(zoo)
```

    ##   animal_name hair feathers eggs milk airborne aquatic preditor toothed
    ## 1    aardvark    1        0    0    1        0       0        1       1
    ## 2    antelope    1        0    0    1        0       0        0       1
    ## 3        bass    0        0    1    0        0       1        1       1
    ## 4        bear    1        0    0    1        0       0        1       1
    ## 5        boar    1        0    0    1        0       0        1       1
    ## 6     buffalo    1        0    0    1        0       0        0       1
    ##   backboned breathes venomous fins legs tails domestic catsize type
    ## 1         1        1        0    0    4     0        0       1    1
    ## 2         1        1        0    0    4     1        0       1    1
    ## 3         1        0        0    1    0     1        0       0    4
    ## 4         1        1        0    0    4     0        0       1    1
    ## 5         1        1        0    0    4     1        0       1    1
    ## 6         1        1        0    0    4     1        0       1    1

``` r
#remove the first column (animal_name)
Data<-zoo[,-1]

#Have a look at how well each class is represented. Is the data set imbalanced?
table(zoo$type)
```

    ## 
    ##  1  2  3  4  5  6  7 
    ## 41 20  5 13  4  8 10

### Preprocess and Tranform Data

Under this section, I need to consider if it is necessary to format, clean, scale etc. the data set. Looking at the table showing the number each class is represented, it's clear that the data set is imbalanced. Class 1 appear way more than the rest of the classes. Also, classes such as 3,4 and 6 are underrepresented. I will use the oversampling method to fix this problem.

``` r
BalanceDataset<-function(Dataset){
  #set all variables to numeric
  Dataset<-sapply(Dataset,as.numeric)
  #frequency table
  n <- table(Dataset[,ncol(Dataset)])
  # convert it to a dataframe
  class<- data.frame(n)
  # Set number of instances for each class equal to the most represented class.
  instances<-max(class$Freq)
  # over sample the dataset such each class is equally represented. 
  along<-as.numeric(class$Var1)
  ind <- unlist(lapply(along, function(i){
    sample(which(Dataset[,ncol(Dataset)]==levels(class$Var1)[i]),size = instances, replace = T)
  }))
  
  #returns a dataset of length = instances*number of classes 
  Dataset[ind,]
}

#Balancing the dataset, such that each class is equally represented in the dataset. 
Data_Balanced <- BalanceDataset(Data)
#having a look at the data after I have balanced it by using the oversampling method.
table(Data_Balanced[,ncol(Data_Balanced)])
```

    ## 
    ##  1  2  3  4  5  6  7 
    ## 41 41 41 41 41 41 41

### Selecting the Training and Test Data

Now I will split the data in two, one used for training and one for evaluating the model. The most common way is to use a 70/30 split, so I will do the same. With larger data sets, one can use a smaller fraction for the test set.

``` r
#Split the data set into training and testing data. 70%/30%
#Selecting the training data
ind <- sample(1:(nrow(Data_Balanced)), size = floor((nrow(Data_Balanced))*0.7),replace = F)
TrainData<- Data_Balanced[ind,]

#Selecting the test data
TestData<- Data_Balanced[-ind,]

#class representation traindata
table(TrainData[,ncol(TrainData)])
```

    ## 
    ##  1  2  3  4  5  6  7 
    ## 28 31 28 28 27 28 30

``` r
#class representation testdata
table(TestData[,ncol(TestData)])
```

    ## 
    ##  1  2  3  4  5  6  7 
    ## 13 10 13 13 14 13 11

Creating the neural network
---------------------------

Now it's time to create the network.

### Neural Network

Making a function to create a neural network and initialize its weights and biases and then use this function to create another function that initializes the population of neural networks.

NeuralNetwork function takes in the following inputs:

-   NI = Number of inputs/numbers of predictors/features
-   NH = number of hiddin layers
-   NprH = number of neurons pr hidden layer
-   NO = number of outputs/ number of classes to predict
-   TrainData = Training data
-   epochs = number of epochs
-   batch\_size = the size of each mini batch
-   lr = learning rate
-   TestData = testing data

The function initializes the weights and biases of the networks and passes them on to the SGD function which is used to train the network.

``` r
NeuralNetwork <- cmpfun(function(NI,NH,NprH,NO,TrainData,epochs,batch_size,lr,TestData){
  
  #the number of neurons in each layer
  struc <- c(NI,rep(NprH,NH),NO)
  
  # number of weights for each neuron
  W_lengths <- struc[1:(length(struc)-1)]
  
  # Bias for each neuron
  B_lengths <- struc[-1]
  
  # initialize the biases
  B <- lapply(seq_along(W_lengths), function(x){
    r <- B_lengths[[x]]
    matrix(rnorm(r,0,1), nrow=r, ncol=1)
  } )

  # initialize weights
  W <- lapply(seq_along(B_lengths), function(x){
    r <- B_lengths[[x]]
    c <-W_lengths[[x]]
    matrix(rnorm(n=r*c,0,1), nrow=r, ncol=c)
  })
  
 out<- SGD(TrainData,epochs,batch_size,lr,TestData,W,B)
  
  return(out)
  
})
```

### Stochastic Gradient Descent (SGD)

The SGD function goes through each epoch and splits the training data into mini batches and updates the weights and biases by using gradient descent and backpropagation to each mini batch

``` r
SGD <- cmpfun(function(TrainData,epochs,batch_size,lr,TestData,W,B){
  # Training the network using stochastic gradient descent. For each epoch, the training data is split into mini batches, and
  #  the weights and biases are updated using gradient descent and backpropagation to the single mini batch.
  
  
  #start timer
  start_time <- Sys.time()

  #check if TestData is providid if yes then the network will evaluated against the TestData each epoch.
if(!missing("TestData"))
  n_test = nrow(TestData)

# storing the number of obs in variable n
n = nrow(TrainData)

# storing the number of mini baches
no_of_mini_batches = floor(n/batch_size)

# iterating thought the epochs
for (i in 1:epochs) {

  #create mini batches
  mini_batches<-lapply(1:no_of_mini_batches, function(i){
                      TrainData[sample(1:nrow(TrainData),size=batch_size,replace = F),]
                        })
  
  # run through each mini batch and update the weigths and biases
  for (j in 1:no_of_mini_batches) {
   mini_bach <- mini_batches[[j]]
   #Applying gradient descent using backpropation to the mini batch
   adj_W_B <- Update_Mini_Batch(mini_bach,lr,W,B)
   #update the weights and biases
   W <- adj_W_B$W
   B <- adj_W_B$B
  }
  
  # printing the progress (only if TestData is provided)
  if(!missing("TestData"))
  cat("Epoch: ",i," of ",epochs,", Accuracy: ", round(Evaluate(W=W, B=B,TestData=TestData,include_table = F)*100,2),"%","\n", sep = "")
}

# Computation time
time_taken <- Sys.time() - start_time

# print the training time
cat("Training complete!","Training time:", time_taken,"\n")

#Returning the Weights and Biases
return(list(W=W,B=B))
})
```

### Update weights and biases for each mini batch

The Update\_Mini\_Batch function updates the weights and biases by applying gradient descent and backpropagation to each mini batch.

``` r
Update_Mini_Batch <- cmpfun(function(batch,lr,W,B){
  
  features <- batch[,-ncol(batch)]
  class = batch[,ncol(batch)]
  
  #initilize list to store bias changes
  B_change <- lapply(seq_along(W), function(x){
    r <- nrow(B[[x]])
    matrix(0, nrow=r, ncol=1)
  } )
  
  #initilize list to store weight changes
  W_change <- lapply(seq_along(W), function(x){
    r <- nrow(W[[x]])
    c <-ncol(W[[x]])
    matrix(0, nrow=r, ncol=c)
  })
  
  #iterate through the mini batch
  for (i in 1:nrow(batch)) {
    x=as.numeric(features[i,])
    #nrow(W[[length(W)]]) = number of classes/outputs
    y<-rep(0,nrow(W[[length(W)]]))
    y[class[i]] <-1
    
    deltas<- Backpropegation(W=W,B=B,x=x,y=y)
    delta_W = deltas$delta_W
    delta_B = deltas$delta_B
   
   # Add gradient from each obs in the mini batch
   B_change<-lapply(seq_along(B), function(j)
     B_change[[j]]+delta_B[[j]]
   )
   
   W_change<-lapply(seq_along(W), function(j)
     W_change[[j]]+delta_W[[j]]
   )
    
  }
  
  # move in the opposite direction of the gradient
  W<-lapply(seq_along(W), function(j)
    # Weights - average gradient across the mini batch * learning rate
    W[[j]] - W_change[[j]]*(lr/nrow(batch))
  )
  B<-lapply(seq_along(B), function(j)
    # Weights - average gradient across the mini batch * learning rate
    B[[j]] - B_change[[j]]*(lr/nrow(batch))
  )
  
  return(list(W=W, B=B))
  
})
```

### Backpropagation

The Backpropagation function computes the gradient of the cost function. The function is consists of 5 parts: 1. Input x: Set the corresponding activation a1 for the input layer 2. Feedforward 3. Output error 4. Backpropagate the error 5. Output

``` r
Backpropegation<- cmpfun(function(W,B,x,y){
  
  #initilize list to store the partial derivatives with respect to the biases
  delta_B <- lapply(seq_along(W), function(x){
    r <- nrow(B[[x]])
    matrix(0, nrow=r, ncol=1)
  } )
  
  #initilize list to store the partial derivatives with respect to the weights
  delta_w <- lapply(seq_along(W), function(x){
    r <- nrow(W[[x]])
    c <-ncol(W[[x]])
    matrix(0, nrow=r, ncol=c)
  })
  
  # Input x: Set the corresponding activation a1 for the input layer
  a <- matrix(x, nrow=length(x), ncol=1)
  #Feedforward
  activations <-list(a)
  Zs<- list()
  for (layer in seq_along(W)){
    b <- B[[layer]]
    w <- W[[layer]]
    w_a <- w%*%a
    bias <- matrix(b, nrow=dim(w_a)[1], ncol=dim(w_a)[-1])
    Z <- w_a + bias
    Zs[[layer]] <-Z
    a <- sigmoid(Z)
    activations[[layer+1]]<-a
  }
  
  
  #Last layer L
  # Output error = (a^(L)-y)%*%sigma'(z^(l))
  delta <- Cost_delta(a=activations[[length(activations)]], y=y) * sigmoid_delta(Zs[[length(Zs)]])
  #The partial derivatives with respect to the biases in output layer
  delta_B[[length(delta_B)]] <- delta
  #The partial derivatives with respect to the weights in output layer
  delta_w[[length(delta_w)]] <- delta%*%t(activations[[length(activations)-1]])
  # Backpropagate the error
  #Layer l
  if (length(W) > 1) {
    for (layer in 2:length(W)) {
      sp<-sigmoid_delta(Zs[[length(Zs)-(layer-1)]])
      # error for the layer
      delta <-(t(W[[length(W)-(layer-2)]])%*%delta) *sp
      #The partial derivatives with respect to the biases
      delta_B[[layer-1]]<-delta
      #The partial derivatives with respect to the weights
      delta_w[[layer-1]]<-delta%*%t(activations[[length(activations)-layer]])
    }
  }
  # Output
  #Return gradient
  gradient = list(delta_W=delta_w,delta_B=delta_B)
  return(gradient)
})
```

#### Helper Functions to Backpropagation

``` r
# Activation function
sigmoid <-function(z){
  return(1/(1+exp(-z)))
}

# the derivative of the activation function
sigmoid_delta<-function(z){
  return(sigmoid(z)*(1-sigmoid(z)))
}
# The derivative of the cost function
Cost_delta <- function(a,y){
  return(a-y) 
}
```

### Evaluation of the model

The function is used to evaluate the performance of the network. The Evaluate function is called in the SGD function if test data is provided.

``` r
# input a network or weights and biases + testdata
Evaluate <-function(Neuralnet,W,B, TestData, include_table=T){
  
  if(missing("Neuralnet")){
   W=W
   B=B
  }else{
    W=Neuralnet$W
    B=W=Neuralnet$B
  }
  
  result <-t(sapply(1:nrow(TestData), function(i){
    a=TestData[i,-ncol(TestData)]
    pred <- Feedforward(W,B,a)
    actual <- TestData[i,ncol(TrainData)]
    matrix(c(pred,actual))
  }))
  
 # Accuracy 
Acc<-  sum(ifelse(result[,1]==result[,2],1,0))/nrow(result)
  

if(include_table==T){
  # Confusion matrix
  CM<- table(result[,1],result[,2])
   return(list(Confusion_matrix = CM, Accuracy = Acc))
}else{
  return(Acc) 
}
}


#Feedforward - Helper function to the Evaluate function
Feedforward <- function(W,B,a){
  
    for (i in seq_along(W)){
      a <- matrix(a, nrow=length(a), ncol=1)
      b <- B[[i]]
      w <- W[[i]]
      w_a <- w%*%a
      bias <- matrix(b, nrow=dim(w_a)[1], ncol=dim(w_a)[-1])
      a <- sigmoid(w_a + bias)
    }
    return(which.max(a))
}
```

### Train and Evaluate Model

``` r
t<-NeuralNetwork(16,1,10,7,TrainData, epochs =  75, batch_size= 20, lr = 2.5,TestData)
```

    ## Epoch: 1 of 75, Accuracy: 40.23%
    ## Epoch: 2 of 75, Accuracy: 54.02%
    ## Epoch: 3 of 75, Accuracy: 57.47%
    ## Epoch: 4 of 75, Accuracy: 58.62%
    ## Epoch: 5 of 75, Accuracy: 60.92%
    ## Epoch: 6 of 75, Accuracy: 60.92%
    ## Epoch: 7 of 75, Accuracy: 65.52%
    ## Epoch: 8 of 75, Accuracy: 66.67%
    ## Epoch: 9 of 75, Accuracy: 62.07%
    ## Epoch: 10 of 75, Accuracy: 62.07%
    ## Epoch: 11 of 75, Accuracy: 66.67%
    ## Epoch: 12 of 75, Accuracy: 66.67%
    ## Epoch: 13 of 75, Accuracy: 65.52%
    ## Epoch: 14 of 75, Accuracy: 66.67%
    ## Epoch: 15 of 75, Accuracy: 65.52%
    ## Epoch: 16 of 75, Accuracy: 77.01%
    ## Epoch: 17 of 75, Accuracy: 68.97%
    ## Epoch: 18 of 75, Accuracy: 75.86%
    ## Epoch: 19 of 75, Accuracy: 75.86%
    ## Epoch: 20 of 75, Accuracy: 77.01%
    ## Epoch: 21 of 75, Accuracy: 77.01%
    ## Epoch: 22 of 75, Accuracy: 82.76%
    ## Epoch: 23 of 75, Accuracy: 79.31%
    ## Epoch: 24 of 75, Accuracy: 83.91%
    ## Epoch: 25 of 75, Accuracy: 89.66%
    ## Epoch: 26 of 75, Accuracy: 83.91%
    ## Epoch: 27 of 75, Accuracy: 83.91%
    ## Epoch: 28 of 75, Accuracy: 83.91%
    ## Epoch: 29 of 75, Accuracy: 96.55%
    ## Epoch: 30 of 75, Accuracy: 93.1%
    ## Epoch: 31 of 75, Accuracy: 98.85%
    ## Epoch: 32 of 75, Accuracy: 96.55%
    ## Epoch: 33 of 75, Accuracy: 100%
    ## Epoch: 34 of 75, Accuracy: 96.55%
    ## Epoch: 35 of 75, Accuracy: 96.55%
    ## Epoch: 36 of 75, Accuracy: 96.55%
    ## Epoch: 37 of 75, Accuracy: 96.55%
    ## Epoch: 38 of 75, Accuracy: 96.55%
    ## Epoch: 39 of 75, Accuracy: 96.55%
    ## Epoch: 40 of 75, Accuracy: 96.55%
    ## Epoch: 41 of 75, Accuracy: 96.55%
    ## Epoch: 42 of 75, Accuracy: 96.55%
    ## Epoch: 43 of 75, Accuracy: 96.55%
    ## Epoch: 44 of 75, Accuracy: 96.55%
    ## Epoch: 45 of 75, Accuracy: 96.55%
    ## Epoch: 46 of 75, Accuracy: 96.55%
    ## Epoch: 47 of 75, Accuracy: 96.55%
    ## Epoch: 48 of 75, Accuracy: 96.55%
    ## Epoch: 49 of 75, Accuracy: 96.55%
    ## Epoch: 50 of 75, Accuracy: 100%
    ## Epoch: 51 of 75, Accuracy: 96.55%
    ## Epoch: 52 of 75, Accuracy: 96.55%
    ## Epoch: 53 of 75, Accuracy: 96.55%
    ## Epoch: 54 of 75, Accuracy: 96.55%
    ## Epoch: 55 of 75, Accuracy: 100%
    ## Epoch: 56 of 75, Accuracy: 96.55%
    ## Epoch: 57 of 75, Accuracy: 100%
    ## Epoch: 58 of 75, Accuracy: 96.55%
    ## Epoch: 59 of 75, Accuracy: 96.55%
    ## Epoch: 60 of 75, Accuracy: 96.55%
    ## Epoch: 61 of 75, Accuracy: 96.55%
    ## Epoch: 62 of 75, Accuracy: 100%
    ## Epoch: 63 of 75, Accuracy: 100%
    ## Epoch: 64 of 75, Accuracy: 96.55%
    ## Epoch: 65 of 75, Accuracy: 100%
    ## Epoch: 66 of 75, Accuracy: 96.55%
    ## Epoch: 67 of 75, Accuracy: 100%
    ## Epoch: 68 of 75, Accuracy: 100%
    ## Epoch: 69 of 75, Accuracy: 100%
    ## Epoch: 70 of 75, Accuracy: 100%
    ## Epoch: 71 of 75, Accuracy: 100%
    ## Epoch: 72 of 75, Accuracy: 100%
    ## Epoch: 73 of 75, Accuracy: 100%
    ## Epoch: 74 of 75, Accuracy: 100%
    ## Epoch: 75 of 75, Accuracy: 100%
    ## Training complete! Training time: 2.766692

``` r
Evaluate(W=t$W, B= t$B,TestData = TestData)
```

    ## $Confusion_matrix
    ##    
    ##      1  2  3  4  5  6  7
    ##   1 13  0  0  0  0  0  0
    ##   2  0 10  0  0  0  0  0
    ##   3  0  0 13  0  0  0  0
    ##   4  0  0  0 13  0  0  0
    ##   5  0  0  0  0 14  0  0
    ##   6  0  0  0  0  0 13  0
    ##   7  0  0  0  0  0  0 11
    ## 
    ## $Accuracy
    ## [1] 1
