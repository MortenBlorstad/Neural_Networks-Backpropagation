rm(list =ls(all=t))
set.seed(1)
library(readr)
library(compiler)

#=========================================================
# Data
#=========================================================

setwd("C:/Users/mob/OneDrive - Norwegian Hull Club/Utdanning/Neural Network - Backpropagation")

#import dataset
zoo <- read.csv("zoo.txt", header=T)

# have a look at the data structure
str(zoo)
head(zoo)

#remove the first column (animal_name)
Data<-zoo[,-1]

#Have a look at how well each class is represented. Is the data set imbalanced?
table(zoo$type)

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

#Selecting the training data
ind <- sample(1:(nrow(Data_Balanced)), size = floor((nrow(Data_Balanced))*0.7),replace = F)
TrainData<- Data_Balanced[ind,]

#Selecting the test data
TestData<- Data_Balanced[-ind,]

#class representation traindata
table(TrainData[,ncol(TrainData)])

#class representation testdata
table(TestData[,ncol(TestData)])

#=========================================================
# End Data
#=========================================================


#=========================================================
# Creating the neural network
#=========================================================

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
#=========================================================





#=========================================================
# Stochastic Gradient Descent (SGD) 
#=========================================================

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

#=========================================================


#=========================================================
# Update weights and biases for each mini batch
#=========================================================
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
#=========================================================

#=========================================================
# Helper functions used in backpropagation 
#=========================================================
# Activation function
sigmoid <-function(z){
  return(1/(1+exp(-z)))
}

sigmoid_delta<-function(z){
  return(sigmoid(z)*(1-sigmoid(z)))
}

Cost_delta <- function(a,y){
  return(a-y) 
}
#=========================================================

#=========================================================
# Backpropagation 
#=========================================================

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
  
  #Feedforward
  a <- matrix(x, nrow=length(x), ncol=1)
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
  
  # propegate backwards
  #Last layer
  # error for output layer = (a^(L)-y)%*%sigma'(z^(l))
  delta <- Cost_delta(a=activations[[length(activations)]], y=y) * sigmoid_delta(Zs[[length(Zs)]])
  #The partial derivatives with respect to the biases in output layer
  delta_B[[length(delta_B)]] <- delta
  #The partial derivatives with respect to the weights in output layer
  delta_w[[length(delta_w)]] <- delta%*%t(activations[[length(activations)-1]])
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
  
  #Return gradient
  deltas = list(delta_W=delta_w,delta_B=delta_B)
  return(deltas)
})

#=========================================================

#=========================================================
# Evaluation of the model
#=========================================================

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


#Feedforward
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


#=========================================================

#=========================================================
# Train model and evaluate
#=========================================================

t<-NeuralNetwork(16,1,10,7,TrainData, epochs =  75, batch_size= 20, lr = 2.5,TestData)

Evaluate(W=t$W, B= t$B,TestData = TestData)
