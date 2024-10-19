import pandas as pd
from flask import Flask, render_template, request
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pickle

app = Flask(__name__)

# Load the dataset and train the Decision Tree model
def train_decision_tree():
    data = pd.read_csv('creditcard.csv')
    X = data[['Amount']]  # Use only the 'Amount' column
    Y = data['Class']     # Class column (0 = valid, 1 = fraud)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Train the Decision Tree Classifier
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    
    # Save the model
    with open('models/decision_tree_model.pkl', 'wb') as file:
        pickle.dump(clf, file)

# Load the pre-trained Decision Tree model
def load_model():
    with open('models/decision_tree_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Train the model when starting the app (you can comment this out if already trained)
train_decision_tree()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/decision-tree')
def decision_tree():
    return render_template('decision_tree.html')

@app.route('/decision-tree-result', methods=['POST'])
def decision_tree_result():
    # Get the amount input from the form
    amount = float(request.form['amount'])
    
    # Prepare the input for the model (reshape as 2D array)
    X_input = [[amount]]
    
    # Load the pre-trained model
    decision_tree_model = load_model()
    
    # Predict using the model
    prediction = decision_tree_model.predict(X_input)
    
    # Result: Fraud (1) or Valid (0)
    result = "Fraudulent" if prediction[0] == 1 else "Valid"
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
