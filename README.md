# Sentiment Analysis


In this project, we will delve into three Natural Language Processing (NLP) models, focusing specifically on sentiment analysis. We will explore three distinct architectures: Fully Connected Neural Network (FCNN), Long Short-Term Memory (LSTM), and finally, the Transformers architecture using the BERT model.

The main objectives of this project are as follows:

- Evaluate the performance of the three models in relation to the sentiment analysis task.
- Assess the level of complexity involved in building each architecture.
- Compare the training times required for each model. 

Upon completion of this study, we aim to draw informed conclusions about which model proves most suitable for sentiment analysis. To achieve this, we will employ a dataset with six classes for sentiment classification.<br>
In all three projects we will use the same database but manipulate it with different techniques.


# Model 1: Fully Connected Neural Network (FCNN)


In this first model we will address the FCNN architecture, A Fully Connected Neural Network (FCNN) is a foundational architecture in deep learning where each neuron in one layer is connected to every neuron in the subsequent layer. It consists of input, hidden, and output layers, with neurons applying weighted sums and activation functions to produce outputs. FCNNs are versatile for tasks like classification and regression but may struggle with sequential data processing. Despite this, they remain valuable in scenarios with fixed-size input data or less critical sequential information.

<img src="https://cdn.discordapp.com/attachments/809675955689881640/1227716593791012944/FCNN.png?ex=66296b2d&is=6616f62d&hm=ba82218c2327e8f4e99738e6b7756ffb30156e07f87cae36078807bb327cfbcf&" alt="Fully Connected Neural Network (FCNN)">

## Techniques used for data processing in the FCNN model

- ## Spacy
In all models we will use Spacy as a way to simplify the text by removing very repetitive words.
- ## TF-IDF:
The TfidfVectorizer from the scikit-learn library, which is a tool used to convert a collection of raw documents into a TF-IDF feature matrix(Term Frequency-Inverse Document Frequency). TF-IDF is a statistical technique used to quantify the importance of a word in a set of documents, commonly used in natural language processing and information retrieval tasks.
- ## Label Encoder
Label Encoder is a data preprocessing technique that converts categorical variables into numeric values. This is useful so that Machine Learning algorithms, which generally work with numbers, can process these variables.
- ## compute_class_weight
This is a scikit-learn function that calculates weights for classes. These weights can be used in classification models to give more importance to classes that are underrepresented in the dataset.
- ## Callbacks end Early Stopping
Another technique that will also be used in the three models, Callbacks end Early Stopping are functions that allow you to monitor and control the training process of machine learning models. They provide a communication channel between the model and the user code, allowing the execution of personalized actions at specific points in the training, such as adjusting the learning rate function and stopping training after the model reaches its plateau.

## Results

<img src="https://cdn.discordapp.com/attachments/809675955689881640/1227716594013175949/result_model_v1.png?ex=6641ceed&is=66407d6d&hm=17a2f4a94d7107643b099b3eefd32d9e1f18777f7cb40d3c3e34324d0f1b297d&" alt="Model FCNN"><br>
<img src="https://cdn.discordapp.com/attachments/809675955689881640/1239270020434100345/image.png?ex=66424fa5&is=6640fe25&hm=04cc587daaa1a2377b557cefdeb5fdafbd6a20b2cae01469fc7b26aa0d0c83c3&" alt="Model FCNN"><br>

- We were able to achieve an accuracy of 83% and it only takes us 2 minutes for training. We'll see how the other architectures fare.

<br>

## If you want to reproduce the experiment on your machine, below are the versions used

<img src="https://cdn.discordapp.com/attachments/809675955689881640/1227734807627173938/image.png?ex=66297c24&is=66170724&hm=1762e9a8a1238c8f981d2e78ddd6ab538bf4f12b31d63a1ecd702ea49eed83f4&" alt="version"><br>

<br>
<br>


- Link to the second model

<a href="https://github.com/CoyoteColt/Sentiment-Analysis-LSTM">Model 2 - Long Short-Term Memory-LSTM</a>

