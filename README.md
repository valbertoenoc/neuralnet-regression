# neuralnet-regression
Neural Network used on a regression problem.

<h4> Problem characterization </h4>

The problem proposes the generation of a model to fit a line on a specific amount of data points. <br>
In that regard, we have 10 input parameters (features) and 2 output parameters (slope and intercept) of the desired line equation.
Since the output parameters are continous and non-categorical, this problem is charecterized as a <strong> Regression Problem </strong>. 

Each sample entry corresponds to 10 pair of coordinates (x,y), or data points, that can be modeled by a single straight line. Thus making it a linear problem if isolated to each data row. However, different roles presents a very distinct set of data points, requiring a completely different line to adjust that particular entry of points. That accounts for the non-linearity of the problem. 

<h4> Neural Networks </h4>

Neural Networks are powerful supervised learning algorithms, that can be used for either Classification or Regression problems. This algorithm can be structured in layers of neurons in such way that the <strong>input layer</strong> accounts for amount of features need to model a data, an arbitrary amount of <strong> Hidden Layers </strong> to are responsible for feature selecting, and hyper space generation, and it accounts for solving non-linear problems due to complex data propagation between neurons, and finally, an <strong> output layer </strong> corresponding to the variables the will be predicted or classified.

In that context, we can roughly state that the hidden layers are responsible for each line fitted to the data, since there are a great amount of distinct set of points to be fitted, more hidden layers become necessary, to which point the network is called a <strong> deep neural network </strong> 

<h4> Code </h4>

This code implements a Multi Layer Perceptron Regressor with three hidden layers using Sklearn Machine Learning toolkit in order to analyse this algorithm performance on this specific problem. 

All the parameters ware empirically selected, requiring further study of feature selection and scaling in order to improve results.

This problem more probably is inserted in the deep learning category for its complex non-linearity and thus could not be solved in a satisfactory manner by the implemented Neural Network.
