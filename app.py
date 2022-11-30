# Importing essential libraries
import numpy as np
from flask import Flask, render_template, jsonify, request
import pickle
import pandas as pd
import matplotlib.pyplot as plt

app = Flask(__name__)
model = pickle.load(open('movie_genre.pkl','rb'))



@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['GET'])
def predict():
  dataset = pd.read_csv('kaggle_movie_train.csv')
  import nltk
  import re
  nltk.download('stopwords')
  from nltk.corpus import stopwords
  from nltk.stem.porter import PorterStemmer
  corpus = []
  ps = PorterStemmer()

  for i in range(0, dataset.shape[0]):

    # Cleaning special character from the dialog/script
    dialog = re.sub(pattern='[^a-zA-Z]', repl=' ', string=dataset['text'][i])

    # Converting the entire dialog/script into lower case
    dialog = dialog.lower()

    # Tokenizing the dialog/script by words
    dialog = dialog.split()

    # Removing the stop words
    dialog = [ps.stem(word) for word in dialog if not word in set(stopwords.words('english'))]


    # Joining the stemmed words
    dialog = ' '.join(dialog)

    # Creating a corpus
    corpus.append(dialog)
  
  from sklearn.feature_extraction.text import CountVectorizer
  cv = CountVectorizer(max_features=10000, ngram_range=(1,2))
  X = cv.fit_transform(corpus).toarray()
  import re
  dialog = re.sub('[^a-zA-Z]', ' ', str('text'))
  dialog=dialog.lower()
  print(dialog)
  import nltk
  nltk.download('stopwords')
  from nltk.corpus import stopwords
  dialog = dialog.split()
  print(dialog)
  dialog1 = [word for word in dialog if not word in set(stopwords.words('english'))]
  print(dialog1)
  from nltk.stem.porter import PorterStemmer
  ps = PorterStemmer()
  dialog = [ps.stem(word) for word in dialog1 if not word in set(stopwords.words('english'))]
  print(dialog)
  dialog2 = ' '.join(dialog)
  print(dialog2)

  X = cv.transform(dialog).toarray()
  input_pred = model.predict(X)
  input_pred = input_pred.astype(float)
  print(input_pred)
  if input_pred[0]==1:
    result = "Action"
  elif input_pred[0]==2:
    result = "Adventure"
  elif input_pred[0]==3:
    result = "Comedy"
  elif input_pred[0]==4:
    result = "Drama"
  elif input_pred[0]==5:
    result = "Horror"
  elif input_pred[0]==6:
    result = "Romance"
  elif input_pred[0]==7:
    result = "Sci-fi"
  elif input_pred[0]==8:
    result = "Thriller"
  elif input_pred[0]==0:
    result = "Miscellaneous"

    result1 = (request.args.get('text'))
    prediction = model.predict([[result1]])
  return render_template('result.html', Result=prediction)

if __name__ == "__main__":
	app.run(debug = True)
