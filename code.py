# Without cross validation original
# Import the libraries
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import numpy as np
import gradio as gr

# Load the dataset
df = pd.read_excel(r'C:\Users\chalw\PycharmProjects\Ideathon\blood sample(altered4).xlsx')

# Separate the features and the target
X = df.drop('Age', axis=1)  # Features
y = df['Age']  # Target

# Create an imputer object to replace NaN values with the mean
imputer = SimpleImputer(strategy='mean')

# Fit the imputer on the features
imputer.fit(X)

# Transform the features with the imputed values
X = imputer.transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the random forest model
rf = RandomForestRegressor(n_estimators=175, max_depth=15, random_state=42)
rf.fit(X_train, y_train)

# Function to make predictions
def predict_age(hemoglobin, platelet_count, wbc_count, rbc_count, lymphocyte_percentage, monocyte_percentage,
                neutrophil_percentage):
    try:
        # Convert input features to a numpy array
        features = np.array([[hemoglobin, platelet_count, wbc_count, rbc_count, lymphocyte_percentage, monocyte_percentage,
                              neutrophil_percentage]])

        # Make predictions using the trained random forest model
        prediction = rf.predict(features)

        # Calculate metrics on the testing set
        y_pred = rf.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Return the predicted age and evaluation metrics
        return f'Predicted biological age: {prediction[0]:.2f} years\n' \
               f'Mean Absolute Error (Test): {mae:.2f}\n' \
               f'Mean Squared Error (Test): {mse:.2f}\n' \
               f'R-squared (Test): {r2:.2f}'

    except Exception as e:
        return f'Error: {str(e)}'

# Define the Gradio interface
iface = gr.Interface(predict_age,
                      inputs=["number", "number", "number", "number", "number", "number", "number"],
                      outputs="text",
                      live=True,
                      title='Biological Age Prediction',
                      theme='huggingface')

# Launch the interface
iface.launch()
