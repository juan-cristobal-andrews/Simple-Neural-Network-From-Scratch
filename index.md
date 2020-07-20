# Introduction

The idea of this notebook is to explore a step-by-step approach to create a <b>single layer neural network</b> without the help of any third party library. In practice, this neural network should be useful enough to generate a simple non-linear regression model, though it's final goal is to help us understand the inner workings of one.

<img src="images/simpleneuralnetwork.jpg" width="495"/>

## 1. Working Data

First we will create a <b>secret function</b> that will generate a test score based on students hours of sleep and study. Note that in real life scenarios not only these secret functions will be unknown but in practice they usually dont exist, meaning, underlying relations between variables such as a Sleep and Study is far more complex and cannot be defined by a simple continuous function.
    
Additionally, as we will later observe, we expect that our neural network should provide us good approximations or predictors of the score but the actual secret function will remain unknown. In other works, we will only have a different, complex continuous function in which its output should be enough to approximate the original one.

```R
# Our secret function
secretFunction <- function(x) {
  y <- (x[,1]^2 + x[,2]^2)
  return(t(t(y)))
}
```

Let's assume a sample of 9 students, where each one had 3 days (72 hours) prior to the test and they either slept or studied.

```R
# Our train (X) and test (xTest) data
Study <- round(runif(9,1,24))
Sleep <- 72 - Study
X <- data.frame(Study=Study,Sleep=Sleep)
xTest = rbind(c(3,7),c(2,8))

# We generate our Y train (y)
y <- secretFunction(X)
```

```R
# This is our Study, Sleep and Score table
cbind(X,Score=y)
```

<img src="images/table.png" width="145" />

## 2. Generating the model

### 2.1 Functions

<b>First, we need some functions to be defined:</b>
- <b>Rand</b>: Generate random numbers
- <b>Sigmoid</b>: Our non-linear activation function to be executed by our Sigmoid neuron.
- <b>Forward</b>: Our forward propagation function.
- <b>Sigmoid Prime</b>: Gradient of our Sigmoid function for Backward Propagation.
- <b>Cost</b>: Cost calculation funtion (sum of squared errors)

```R
# Random Function
rand <- function(x) { 
  return(runif(1, 5, 100)/100) 
}

# Sigmoid Function
sigmoid <- function(x) {
  return(1/(1+exp(1)^-x))
}

# Forward Propagation Function
Forward <- function(X,w1,w2) {
  X <- cbind(X[,1],X[,1])
  z2 <- X %*% w1
  a2 <- sigmoid(z2)
  z3 <- a2 %*% w2
  yHat <- sigmoid(z3)
  return(yHat)
}

# Sigmoid Gradient Function
sigmoidPrime <- function(x) {
  return((exp(1)^-x)/(1+(exp(1)^-x))^2)
}

# Cost Function
cost <- function(y,yHat) {
  sum(((yHat-y)^2)/2)
}
```

### 2.2 Parameter Initialization

Next, we need to define our parameters.
We have two sets of parameters:
- <b>Hyperparameters:</b> Parameters that the network cannot learn and are pre-defined.
    - <b>Number of hidden layers:</b> In this case we have 1, since it's a simple single layered neural network.
    - <b>Number of Neurons of hidden layers:</b> We will use 6.
    - <b>Learning Rate:</b> We will use 2.
- <b>Learning Parameters:</b> Parameters that our network will <b>learn</b>.
    - <b>Weights:</b> We will use 2 weights since by design we will need at leas N+1 weights where N is equivalent to the number of Hidden Layers.

```R
# Hyperparameters
inputLayerSize = ncol(X)
outputLayerSize = 1 # Dimension of outputs (1 since it's only score)
hiddenLayerSize = 6 # Number of neurons
LearningRate <- 2

# Weights
w1 <- matrix(runif(inputLayerSize*hiddenLayerSize), nrow = inputLayerSize, ncol = hiddenLayerSize )
w2 <- matrix(runif(hiddenLayerSize*outputLayerSize), nrow = hiddenLayerSize, ncol = outputLayerSize )
```

### 2.3 Data Normalization

```R
# We normalize train data
X = X/max(X)
y = y/max(y)

# We normnalize test data
xTest <- xTest/max(X)
yTest <- secretFunction(xTest)/max(y)
```

## 3. Propagation

### 3.1 Forward Propagation

```R
# We propagate
yHat <- Forward(X,w1,w2)
```

### 3.1.1 Cost Calculation

```R
# We calculate cost
J <- sum(((yHat-y)^2)/2)
J
```
<img src="images/output1.png" width="146" height="26" />

### 3.1.2 We evaluate the results

```R
library(ggplot2)
resultPlot <- as.data.frame(rbind(cbind(y,1:nrow(y),"Real"),cbind(round(yHat,2),1:nrow(yHat),"Prediccion")))
ggplot(resultPlot, aes(x=V2, y=V1, fill=V3)) + geom_bar(stat="identity", position="dodge")
```

<img src="images/chart1.png" width="420" height="420" />

### 3.2 Back propagation

```R
# We derivate W2 in respect to the cost
dJdW2 <- function(X,w1,w2) { 
  X <- cbind(X[,1],X[,1])
  z2 <- X %*% w1
  a2 <- sigmoid(z2)
  z3 <- a2 %*% w2
  yHat <- sigmoid(z3)
  delta3 <- -(y-yHat)*sigmoidPrime(z3)
  cost <- t(a2) %*% delta3
  return(cost)
}

# We adjust W2
w2 <- w2 - (LearningRate * dJdW2(X,w1,w2))

# We derivate W1 in respect to the cost
dJdW1 <- function(X,w1,w2) { 
  X <- cbind(X[,1],X[,1])
  z2 <- X %*% w1
  a2 <- sigmoid(z2)
  z3 <- a2 %*% w2
  yHat <- sigmoid(z3)
  delta3 <- -(y-yHat)*sigmoidPrime(z3)
  delta2 <- (delta3 %*% t(w2)) * sigmoidPrime(z2)
  cost <- t(X) %*% delta2
  return(cost)
}
w1 <- w1 - (LearningRate * dJdW1(X,w1,w2))
```

### 3.3 We forward propagate again

```R
# We propagate Again!
yHat <- Forward(X,w1,w2)
```

### 3.3.1 New Cost Calculation

```R
# We calculate cost
J <- sum(((yHat-y)^2)/2)
J
```
<img src="images/output2.png" width="144" height="21" />


<b>Note:</b> We should observe a small improvement in cost due to the new parameters.

### 3.3.2 We evaluate again

```R
library(ggplot2)
resultPlot <- as.data.frame(rbind(cbind(y,1:nrow(y),"Real"),cbind(round(yHat,2),1:nrow(yHat),"Prediccion")))
ggplot(resultPlot, aes(x=V2, y=V1, fill=V3)) + geom_bar(stat="identity", position="dodge")
```

<img src="images/chart2.png" width="420" height="420" />

## 4. Backpropagate, Forwardpropagate and repeat

We will now repeat the previous process until we cannot minimize our cost any more.
When this happens, it means we have found a <b>local minima</b>. We will stop when we observe that error calculated at <b>step n+1</b> is equal or superior than the one found in <b>step n</b>, meaning we cannot improve any more with out jumping around the local minima.

```R
costTrain <- data.frame(Training=NA,Cost=NA)
costTest <- data.frame(Training=NA,Cost=NA)
InitialError <- sum((y-yHat)^2)
FinalError <- 0
i <- 1

while(round(FinalError,5) <= round(InitialError,5)) {
  w1 <- w1 - (LearningRate * dJdW1(X,w1,w2))
  w2 <- w2 - (LearningRate * dJdW2(X,w1,w2))
  yHat = Forward(X,w1,w2)
  costo <- cost(y,yHat)

  costTrain[i,]$Training <- i
  costTrain[i,]$Cost <- costo
  
  FinalError <- sum((y-yHat)^2)

  i <- i + 1
  if(i %% 1000==0) {
    # Print on the screen some message
    cat(paste0("Iteration ", i,": ",FinalError,"\n"))
  }
  if(i == 30000) {
      break()
  }
}
```

<img src="images/table2.png" width="320" height="513" />

### 4.1 We evaluate again

```R
library(ggplot2)
resultPlot <- as.data.frame(rbind(cbind(y,1:nrow(y),"Real"),cbind(round(yHat,2),1:nrow(yHat),"Prediccion")))
ggplot(resultPlot, aes(x=V2, y=V1, fill=V3)) + geom_bar(stat="identity", position="dodge")
Improvement <- (InitialError-FinalError)/InitialError
cat(paste("Initial Error: ",InitialError,"
Final Error: ",FinalError,"
Improvement: ",round(Improvement,2)*100,"%
Took ",i," Iterations",sep=""))
```

<img src="images/chart3.png" width="431" height="536" />

As seen in the results above it seems our model was able to predict very similar scores to our "Secret Function", even though the actual model is a mix of a more complex combination of vector products and non-linear functions. This means our new model approximates quite well our actual "Secret Function Model".

## 5. How our training improved our model?

```R
costTrain$Data <- "Train"
ggplot(costTrain, aes(x=Training, y=Cost)) + geom_line()
```

<img src="images/cost.png" width="420" height="420" />

As seen above it seems that there was little cost improvement after 1k iterations.

## 6. Evaluation on known (in sample) Data

```R
Train <- X
```
```R
# Note: this output represents a normalized representation of Study and Sleep
cbind(Train,RealScore=secretFunction(Train),PredictedScore=Forward(Train,w1,w2))
```

<img src="images/table3.png" width="310" height="315" />

<b>Lets translate this to our original scale</b>

```R
X <- data.frame(Study=Study,Sleep=Sleep)
y <- secretFunction(X)
cbind(X,Score=secretFunction(X),Prediction=round(Forward(Train,w1,w2)*max(y)))
```

<img src="images/table4.png" width="214" height="291" />

As expected, it seems our model provide us very good approximations to actual test scores.

## 7. Evaluation on unknown (out of sample) data

Let's evaluate which test score we should expect from a student who <b>studied 16 hours and slept 56</b>

```R
xTrain <- data.frame(Study=16,Sleep=56)
yTrain <- secretFunction(xTrain)
cbind(xTrain,Score=yTrain)
```

<img src="images/table5.png" width="145" height="108" />

<b>What is our predicted score?</b>

```R
as.integer(round(Forward(xTrain/max(X),w1,w2)*max(y)))
```

<img src="images/table5.png" width="19" height="38" />

Seems pretty close to the real expected score (3292)

### Simulation: How our model predicts new data

Lets simulate 72 students, starting from a student who studied 0 hours and slept 72, up to the opposite scenario.

```R
Test <- data.frame(Study=seq(0,72))
Test$Sleep <- 72-Test$Study
Test$Score <- secretFunction(Test)
Test$Prediction <- as.integer(round(Forward(Test/max(X),w1,w2)*max(y)))
Test$SquaredError <- (Test$Score - Test$Prediction)^2
Test
```

<img src="images/table4.png" width="321" height="554" />

## 8. Final Thoughts

<b>Let's see how well our model predicts outside our training space.</b>

```R
ggplot(Test, aes(x=Study, y=sqrt(SquaredError))) +
    geom_line() +
    geom_vline(xintercept=min(X$Study), linetype="dashed", color = "red") +
    geom_vline(xintercept=max(X$Study), linetype="dashed", color = "red")
cat(paste("Training Space Known by model:\n   Min Study Hours:",min(X$Study)),"\n   Max Study Hours:",max(X$Study))
RMSEWithin <- round(sqrt(mean(Test$SquaredError[which(Test$Study >= min(X$Study) & Test$Study <=max(X$Study))])))
RMSEBelow <- round(sqrt(mean(Test$SquaredError[which(Test$Study < min(X$Study))])))
RMSEAbove <- round(sqrt(mean(Test$SquaredError[which(Test$Study > max(X$Study))])))
cat(paste("\n\nAverage Root Mean Squared Error:\n   Below Known Range:",RMSEBelow, "\n   Within Known Range:",RMSEWithin,"\n   Above Known Range:",RMSEAbove))
```

<img src="images/chart4.png" width="432" height="537" />

As observed by the errors from the table and the plot above, it seems that our new function had somewhat better prediction capabilities within the training space which is represented by our vertical lines. As expected, our new model is not able to predict out-of-sample data that falls outside of our training space.

In other words, our model is able to interpolate quite well the approximation of Students score by providing their time of Study and Sleep, in contrast, is not able to extrapolate very well outliers or data outside its training space.






















