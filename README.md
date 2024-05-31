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

<img src="https://cdn.discordapp.com/attachments/1244359723629936793/1244359785508507749/1_VHOUViL8dHGfvxCsswPv-Q.png?ex=665a19db&is=6658c85b&hm=e71c49fcf175fe02c76af8cb134aea602ec7687a92bddf644817b54532ab6dbd&" alt="Fully Connected Neural Network (FCNN)">

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

<img src="https://cdn.discordapp.com/attachments/1244359723629936793/1244370125000413346/Untitled.png?ex=665a237d&is=6658d1fd&hm=6b18f0973ae87f7c9d1c0710c587805a55b5b0cc0be8834def41d5bb4901feb3&" alt="Model FCNN"><br>
<img src="https://cdn.discordapp.com/attachments/1244359723629936793/1244370452025966643/1.png?ex=665a23cb&is=6658d24b&hm=9cda1149a311c1a7d02457e1133069b4438fc9d371babb1b369ea8d6e6d24cd9&" alt="Model FCNN"><br>
<img src="https://cdn.discordapp.com/attachments/1244359723629936793/1244370731496640522/3.png?ex=665a240d&is=6658d28d&hm=b52747a1106b4acf86a486ed11215a53ec53beaca479692033064a4f9d70df9c&" alt="Model FCNN"><br>

- We were able to achieve an accuracy of 83% and it only takes us 2 minutes for training. We'll see how the other architectures fare.

<br>

## If you want to reproduce the experiment on your machine, below are the versions used

<img src="https://cdn.discordapp.com/attachments/1244359723629936793/1244371114491252766/4.png?ex=665a2468&is=6658d2e8&hm=5f411be1aa44be5bad9cd0308a2a14a39b4bb675d4f736d263c94b0e914d03a3&" alt="version"><br>

<br>
<br>


- Link to the second model

<a href="https://github.com/CoyoteColt/Sentiment-Analysis-LSTM">Model 2 - Long Short-Term Memory-LSTM</a>

