import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import lime
import lime.lime_text
import shap
from datasets import load_dataset

# Load a simple text classification dataset (IMDB reviews)
def load_data():
    dataset = load_dataset("imdb", split="train[:2000]")
    texts = dataset["text"]
    labels = dataset["label"]
    return train_test_split(texts, labels, test_size=0.2, random_state=42)

# Train a simple model
def train_model(X_train, y_train):
    # Convert text to features
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    
    # Train a logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)
    
    return model, vectorizer

# Evaluate the model
def evaluate_model(model, vectorizer, X_test, y_test):
    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

# LIME explanation
def explain_with_lime(model, vectorizer, X_test, class_names):
    # Create a LIME explainer
    explainer = lime.lime_text.LimeTextExplainer(class_names=class_names)
    
    # Choose a sample to explain
    idx = 10
    text_instance = X_test[idx]
    
    # Generate explanation
    exp = explainer.explain_instance(
        text_instance, 
        lambda x: model.predict_proba(vectorizer.transform(x)),
        num_features=10
    )
    
    # Display explanation
    print("\n=== LIME Explanation ===")
    print(f"Text: {text_instance[:100]}...")
    print(f"Predicted class: {class_names[model.predict(vectorizer.transform([text_instance]))[0]]}")
    print("\nFeatures contributing to prediction:")
    for feature, weight in exp.as_list():
        print(f"  {feature}: {weight:.4f}")
    
    # Visualize
    plt.figure(figsize=(10, 6))
    exp.as_pyplot_figure()
    plt.tight_layout()
    plt.savefig('lime_explanation.png')
    print("\nLIME visualization saved as 'lime_explanation.png'")

# SHAP explanation
def explain_with_shap(model, vectorizer, X_test, class_names):
    # Create a SHAP explainer
    X_test_vec = vectorizer.transform(X_test[:20])  # Use a subset for demonstration
    
    # Create a function that returns the model's prediction probabilities
    def model_pred(x):
        return model.predict_proba(x)
    
    # Create a SHAP explainer
    explainer = shap.LinearExplainer(model, X_test_vec)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X_test_vec)
    
    # Choose a sample to explain
    idx = 10
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # Display explanation
    print("\n=== SHAP Explanation ===")
    print(f"Text: {X_test[idx][:100]}...")
    print(f"Predicted class: {class_names[model.predict(vectorizer.transform([X_test[idx]]))[0]]}")
    
    # Get non-zero features for this sample
    nonzero_indices = X_test_vec[0].nonzero()[1]
    
    # Get top features by SHAP value magnitude
    shap_for_instance = shap_values[1][0]  # For positive class
    top_indices = np.argsort(np.abs(shap_for_instance))[-10:]
    
    print("\nTop features contributing to prediction:")
    for i in top_indices:
        if i in nonzero_indices:
            print(f"  {feature_names[i]}: {shap_for_instance[i]:.4f}")
    
    # Visualize
    plt.figure(figsize=(12, 6))
    shap.summary_plot(shap_values, X_test_vec, feature_names=feature_names, class_names=class_names, plot_type="bar")
    plt.tight_layout()
    plt.savefig('shap_summary.png')
    print("\nSHAP summary plot saved as 'shap_summary.png'")

def main():
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Train model
    model, vectorizer = train_model(X_train, y_train)
    
    # Evaluate model
    evaluate_model(model, vectorizer, X_test, y_test)
    
    # Explain predictions
    class_names = ["Negative", "Positive"]
    explain_with_lime(model, vectorizer, X_test, class_names)
    explain_with_shap(model, vectorizer, X_test, class_names)
    
    print("\nDone! You can now examine the visualizations to understand how the model makes decisions.")

if __name__ == "__main__":
    main()