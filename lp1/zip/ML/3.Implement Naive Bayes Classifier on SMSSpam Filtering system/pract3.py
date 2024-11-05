import pandas as pd

df = pd.read_csv("SMSSpamCollection", sep = '\t', names = ['label', 'text'])
print(df)

print(df.shape)

import nltk 
sent = "Hello friends! How are you? We will be learning Python today."


from nltk.tokenize import word_tokenize 

from nltk.corpus import stopwords
swords = stopwords.words('english')


from nltk.stem import PorterStemmer
ps = PorterStemmer()

def clean_text(sent):
    tokens = word_tokenize(sent)
    clean = [word for word in tokens 
             if word.isdigit() or word.isalpha()]  
    clean = [ps.stem(word) for word in clean if word not in swords]
    return(clean)
    
clean_text(sent)


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(analyzer=clean_text)

x = df['text']
y = df['label']

x_new = tfidf.fit_transform(x)
print(x.shape)
print(x_new.shape) 

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_new, y, random_state=0, test_size=0.25)

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train.toarray(), y_train)

y_pred = nb.predict(x_test.toarray())

from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=0)
rf.fit(x_train, y_train)

y_pred = rf.predict(x_test)
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

from sklearn.model_selection import GridSearchCV
params = {
        'criterion': ['gini', 'entropy'],
        'max_features': ['sqrt', 'log2'],
        'random_state': [0,1,2,3,4],
        'class_weight': ['balanced','balanced_subsample']
    }

grid = GridSearchCV(rf, param_grid=params, cv = 5, scoring='accuracy')
grid.fit(x_train, y_train)

print(grid.best_estimator_)
