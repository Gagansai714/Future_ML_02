import joblib

def load_models():
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    cat_model = joblib.load('models/category_model.pkl')
    prio_model = joblib.load('models/priority_model.pkl')
    return vectorizer, cat_model, prio_model

def classify_ticket(ticket_text):
    vectorizer, cat_model, prio_model = load_models()
    tfidf_vec = vectorizer.transform([ticket_text])
    
    category = cat_model.predict(tfidf_vec)[0]
    priority = prio_model.predict(tfidf_vec)[0]
    
    return category, priority

if __name__ == "__main__":
    sample_ticket = "URGENT: Password expired for external database access. Failed connection multiple times."
    cat, prio = classify_ticket(sample_ticket)
    
    print(f"Ticket: {sample_ticket}")
    print(f"Predicted Category: {cat}")
    print(f"Predicted Priority: {prio}")