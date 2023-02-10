import streamlit as st
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import semi_supervised
from sklearn import metrics


from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve , roc_auc_score
from sklearn.metrics import precision_score
from sklearn.base import TransformerMixin
st.sidebar.title("Semi-supervised learning")

st.subheader('Assignment 3(TIMG 5301 - Group 8)')
st.markdown(f'<h1 style="color:#00008B;font-size:20px;">{"Raksha, Vindhya, Sahil, Ashwani"}</h1>',unsafe_allow_html=True)

############################### 2. Data understanding and preparation ############################

def load_corpus():
    corpus = pd.read_csv("ExtractedTweets.csv")
    # Change column names to lowercase
    corpus.columns = [col.lower() for col in corpus.columns]
    # Sections of the corpus we want to keep
    return corpus


def load_labels():
    return pd.read_csv("LabelledData.csv")

# 2.c Load the corpus of tweets and convert it to a data frame

dataFrame = load_corpus()
labels = load_labels()


if st.sidebar.checkbox("Show Unlabelled Data"):
    st.subheader("Unlabelled Data")
    st.write(dataFrame)

 #2.d Label a subset of the tweets as positive (idea) or negative (not an idea)

dataFrame['label'] = -1 
for _, row in labels.iterrows():
    dataFrame.loc[row['id'], 'label'] = row['label']



if st.sidebar.checkbox("Show corpus"):
    st.subheader("Corpus")
    st.dataframe(dataFrame)
    
    
counts = dataFrame['label'].value_counts()
fig, ax = plt.subplots(figsize=(6.4, 2.4))
sns.barplot(x=counts.index, y=counts.values, ax=ax)
ax.set_xticklabels([-1,0,1])
ax.set_xlabel('Tweets')
ax.set_ylabel('Number of Tweets')
ax.set_title('Class imbalance')

Good = counts[1]
Bad = counts[0]

#2.e Determine the ratio of negative to positive tweets. Is the dataset balanced or imbalanced?

if st.sidebar.checkbox('Check imbalance'):
    st.subheader('Class imbalance')
    st.pyplot(fig)
    st.write(f'<h5>{"Degree of imbalance"}</h5>',unsafe_allow_html=True)

    st.write('Positive ideas: ' + str(Good))
    st.write('Negative ideas: '+ str(Bad))
    bad_r = Bad/Good
    st.write('Ratio of negative to positive ideas in the labelled data: ' , round(bad_r, 2))
    
#2.f Definition of stop words.    
    
user_stopwords = st.sidebar.text_area("Stopwords (one per line)",
    on_change=lambda: st.session_state.clear())
@st.cache(allow_output_mutation=True)
def read_stopwords(file):
    file = open(file, 'r')
    return [w.strip() for w in file.read().split('\n')]
additionalStopwords = ["pollution","http"," idea","https","pollutions","ideas"]

#2.g Create a term-document matrix that shows the terms used as features and their TF-IDF values in each document

from sklearn.feature_extraction.text import CountVectorizer
preprocessor = Pipeline([
    ('vectorizer', CountVectorizer(stop_words=['english','additionalStopwords','user_stopwords'], max_features=100, min_df=2)),
    ('tfidf', TfidfTransformer())])
tfidf = preprocessor.fit_transform(dataFrame['content'])
vectorizer = preprocessor.named_steps['vectorizer']
vocab = vectorizer.get_feature_names_out()
tdm = pd.DataFrame(tfidf.toarray().T, index=vocab)



if st.sidebar.checkbox('Show term-document matrix'):
    st.subheader('Term-document matrix')
    st.write('The rows are words and the columns are documents.')
    st.dataframe(tdm)
    
#######################################################3. Modelling ############################################

# 3.a Create a base estimator using the Logistic Regression classifier

#3.b Define a self-training semi-supervised classifier that uses the base estimator

#3.c Create a pipeline that preprocesses the tweets

# Self-training requires a base estimator
base_estimator = LogisticRegression(penalty='l2', class_weight='balanced')
classifier = semi_supervised.SelfTrainingClassifier(base_estimator)
model = Pipeline(steps=[
    ('vectorizer', CountVectorizer(stop_words='english', min_df=2, max_features=100)),
    ('tfidf', TfidfTransformer()),
    ('classifier', classifier)])

class DenseTransformer(TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()
    
def trainedModel(model, dataFrame):
    tmodel = model.fit(dataFrame['content'], dataFrame['label'])
    
    return tmodel
    
def predict(dataFrame, model):
    dataFrame['predicted'] = model.predict(dataFrame['content'])
    
    return dataFrame

model.fit(dataFrame['content'], dataFrame['label'])

#3.e Apply the pipeline to the corpus and show the predicted labels
if st.sidebar.checkbox("Predicted labels"):
    st.subheader("Predicted labels")
    # Predict the labels for the unlabeled tweets
    dataFrame['predicted'] = model.predict(dataFrame['content'])
    

    st.dataframe(dataFrame)
    # Statistics
    st.write("This is a count of the predicted labels:")
    st.write(dataFrame['predicted'].value_counts())

 #3.f Save the model to a file, and create a second Streamlit application that loads the model you saved, and asks the user to
#enter a tweet and reports back the predicted label.
 
filename = "model_file.pkl"
pickle.dump(model,open(filename,'wb'))
if st.sidebar.checkbox("Save the model"):
    st.subheader("Save the model")
    # Predict the labels for the unlabeled tweets    
    # save the model to disk
    st.write("Great! the file successfully got Saved ", filename)
 
 ######################################## 4.Evaluation and interpretation #############################################
 
 
 
if st.sidebar.checkbox("Evaluate the Model"):
    st.subheader("Accuracy")
    accuracy = accuracy_score(dataFrame.loc[0:99,'label'], dataFrame.loc[0:99,'predicted'])
    st.write(accuracy)    
    st.subheader("Precision/Recall/f1-Score")
    #precision = precision_score(corpus['label'], corpus['predicted'], average= None)
    st.text('Model Report:\n ' + classification_report(dataFrame['label'] , dataFrame['predicted']))
    
 #4.a Use stratified k-fold cross-validation to evaluate the model.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_validate(model,dataFrame.loc[0:99,'content'], dataFrame.loc[0:99,'label'], cv=cv, n_jobs=-1,
    scoring=['accuracy', 'roc_auc', 'precision', 'recall', 'f1'])

if st.sidebar.checkbox("Show model performance Stratified k-fold"):
    st.subheader('Model performance Stratified k-fold')
    st.write("The table shows the performance of the logistic regression classifier on the training data.")
 # 4.c  Performance scores for each fold
    st.write("Scores for each fold (only positive class):")
    data_scores = pd.DataFrame(scores).transpose()
    data_scores['mean'] = data_scores.mean(axis=1)
    st.dataframe(data_scores)
    
