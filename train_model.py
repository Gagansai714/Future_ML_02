import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

def assign_priority(text, topic):
    """Rule-based priority assignment for target generation."""
    text_lower = str(text).lower()
    high_keywords = ['urgent', 'expire', 'failed', 'block', 'warning', 'error', 'critical', 'immediately', 'down']
    low_keywords = ['meeting', 'general', 'info', 'inquiry', 'thanks', 'question']

    if any(k in text_lower for k in high_keywords) or topic in ['Access', 'Administrative rights']:
        return 'High'
    elif any(k in text_lower for k in low_keywords) or topic in ['Miscellaneous', 'HR Support']:
        return 'Low'
    return 'Medium'

def main():
    os.makedirs('models', exist_ok=True)
    
    # Load dataset
    df = pd.read_csv('all_tickets_processed_improved_v3.csv')
    df.dropna(subset=['Document', 'Topic_group'], inplace=True)
    
    # Feature engineering for priority
    df['Priority'] = df.apply(lambda row: assign_priority(row['Document'], row['Topic_group']), axis=1)

    X = df['Document']
    y_category = df['Topic_group']
    y_priority = df['Priority']

    # Train/Test Split
    X_train, X_test, y_cat_train, y_cat_test, y_prio_train, y_prio_test = train_test_split(
        X, y_category, y_priority, test_size=0.2, random_state=42, stratify=y_category
    )

    # Vectorization
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train Category Model
    cat_model = LogisticRegression(max_iter=1000)
    cat_model.fit(X_train_tfidf, y_cat_train)
    y_cat_pred = cat_model.predict(X_test_tfidf)

    # Train Priority Model
    prio_model = LogisticRegression(max_iter=1000)
    prio_model.fit(X_train_tfidf, y_prio_train)
    y_prio_pred = prio_model.predict(X_test_tfidf)

    # Evaluation
    print("=" * 50)
    print(f"Category Model Accuracy: {accuracy_score(y_cat_test, y_cat_pred):.4f}")
    print(classification_report(y_cat_test, y_cat_pred))
    
    print("=" * 50)
    print(f"Priority Model Accuracy: {accuracy_score(y_prio_test, y_prio_pred):.4f}")
    print(classification_report(y_prio_test, y_prio_pred))

    # Save artifacts
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
    joblib.dump(cat_model, 'models/category_model.pkl')
    joblib.dump(prio_model, 'models/priority_model.pkl')
    print("Models saved successfully in 'models/' directory.")

if __name__ == "__main__":
    main()