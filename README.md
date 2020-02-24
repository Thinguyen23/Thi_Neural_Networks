# Neural Networks and Deep Learning Models
In this project, I build a machine learning model to predict the success of charity ventures paid by Alphabet soup using TensorFlow library and Keras module.

The complete notebook to the project can be found [here](https://github.com/Thinguyen23/Thi_Neural_Networks/blob/master/AlphabetSoupChallenge.ipynb)
## Data Preprocesing
Starting with the [Alphabet Soup Charity dataset](https://github.com/Thinguyen23/Thi_Neural_Networks/blob/master/charity_data.csv), I first preprocess the dataset by
- filtering out non-feature variables ("NAME" and "STATUS")
- Categorical binning for rare categorical values ("APPLICATION_TYPE" and "CLASSIFICATION")
- Using OneHotEncoder to encode categorical variables
- Scaling the data using StandardScaler after spliting the target(y) and features(X) into training and testing set
## Compile, Train and Evaluate Model
I first create a Keras Sequential model : `nn = tf.keras.models.Sequential()`
Then add hidden layers and number of neurons per layer using Dense class, then the output layer before compiling all the layers in the model. The model is then trained on X_train_scaled and y_train.
## Results
When evaluate the model using test data, the model yield a Loss of 0.5567 and Accuracy of 0.73, which is not optimal but still acceptable.
## Written Anlysis
#### 1. How many neurons and layers did you select for your neural network model?
...The model that I chose has 3 hidden layers with 100 neurons on the first 2 layers and 50 neurons for the last layer
#### 2. Were you able to achieve target model performance?
...I was not able to achieve the target predictive accuracy higher than 75%. My best accuracy was 73%
#### 3. What steps did you take to try and increase model performance?
...I tried several different strategies to increase the model performane including:
- increase the number of neurons per layer to 500
- increase number of hidden layers to 6
- switch activation functions among "relu", "sigmoid" and "tanh"
- increase epochs to 100
...However, after running the model for a while, performance decreased instead of increase. The final model is my best guess on the test data so far.

#### 4. If you were to implement a different model to solve this classification problem, which would you choose and why?
...I would probably go with Random forest classifier. Random forest models use a number of weak learner algorithms and combine their output to make a final decision. The model is robust and can easily handle outliers and nonlinear tabular data, which is the same with the charity dataset we're dealing with.
