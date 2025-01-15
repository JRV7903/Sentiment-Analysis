import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

df = pd.read_csv('path/to/your/dataset.csv')

df.dropna(subset=['text', 'label'], inplace=True)

le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

class_counts = df['label'].value_counts()
if class_counts.min() < class_counts.max():
    df_minority = df[df['label'] == class_counts.idxmin()]
    df_majority = df[df['label'] == class_counts.idxmax()]

    df_minority_upsampled = resample(df_minority, 
                                     replace=True,    
                                     n_samples=len(df_majority),   
                                     random_state=42) 

    df = pd.concat([df_majority, df_minority_upsampled])

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.7)),
    ('logreg', LogisticRegression(max_iter=1000, class_weight='balanced'))  # Adding class_weight to handle imbalance
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

cv_scores = cross_val_score(pipeline, df['text'], df['label'], cv=5)
print(f"Cross-Validation Accuracy: {cv_scores.mean()} ± {cv_scores.std()}")
