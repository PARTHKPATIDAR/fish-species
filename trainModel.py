import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load your fish dataset (replace 'fish_data.csv' with the actual path to your dataset file)
data = pd.read_csv('https://raw.githubusercontent.com/PARTHKPATIDAR/fish-species/main/Fish.csv')

# Prepare the features and target
X = data.drop(columns=['Species'])
y = data['Species']

# Initialize the Decision Tree classifier
model = DecisionTreeClassifier()

# Fit the model on the entire dataset
model.fit(X, y)

# Save the model to a file
joblib.dump(model, 'fish_species_model.pkl')
