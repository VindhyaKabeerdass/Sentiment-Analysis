   **Sentiment-Analysis-Twitter**


**Objective:**

The aim of this project is to analyze the consumers interest and trends and assess the tweets as positive or negative ideas based on the ideas or thoughts the tweet imposes

**Data Sources:**

Approximately 1000 tweets was scrapped from Twitter and around 10% of tweets were labbeled positive negative based on the ideas they impose for training the model. 

**Methodology:**

Semi-supervided model was used to label the remaining tweets

Stopwords were defined to reduce noise in the data and improve teh accuracy of text classification

Base estimator was created 

A logistic regression model is built that uses the base esitmator 

Pipleine was created to display the predicted labels as positive and negative  

Streamlit was used to deploy the code

A interactive application was created to assess the given tweet as positive or negative based on the idea it imposes 

**Conclusion:**

Evaluated the model through stratified k-fold algorithm and achieved 84 % accuracy
