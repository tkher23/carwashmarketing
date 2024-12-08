import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import chardet
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import tensorflow
import ast
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error



# Example DataFrame
# Replace 'your_file.csv' with the path to your CSV file
file_path = 'merged.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

print(df['color_name'].unique())

# 1. Text Vectorization for Post Caption
#tfidf = TfidfVectorizer(max_features=10)  # Adjust features as needed
#df_tfidf = tfidf.fit_transform(df["post_caption"]).toarray()
#df_tfidf = pd.DataFrame(df_tfidf, columns=tfidf.get_feature_names_out())

# 2. Sentiment Polarity: Already normalized (0-1)

# Define reference colors
reference_colors = {
    "Red": [255, 0, 0],
    "Green": [0, 255, 0],
    "Blue": [0, 0, 255],
    "Yellow": [255, 255, 0],
    "Cyan": [0, 255, 255],
    "Magenta": [255, 0, 255],
    "White": [255, 255, 255],
    "Black": [0, 0, 0]
}

# Convert reference colors to numpy array
ref_color_names = list(reference_colors.keys())
ref_color_values = np.array(list(reference_colors.values()))

# Function to find nearest reference color
def assign_nearest_color(rgb):
    rgb = np.array(rgb)
    distances = np.linalg.norm(ref_color_values - rgb, axis=1)
    nearest_color_index = np.argmin(distances)
    return ref_color_names[nearest_color_index]

# Apply the function
df["average_color"] = df["average_color"].apply(ast.literal_eval)

df["color_group"] = df["average_color"].apply(assign_nearest_color)

# One-hot encode the Color_Group column
one_hot_encoder = OneHotEncoder(sparse_output=False)
encoded_colors = one_hot_encoder.fit_transform(df[["color_group"]])

# Add encoded columns to the dataset
color_columns = one_hot_encoder.get_feature_names_out(["color_group"])
df_encoded = pd.DataFrame(encoded_colors, columns=color_columns)

# Combine with original dataset
df = pd.concat([df, df_encoded], axis=1)

#Hashtag
df['hashtag_binary'] = df['Hashtags? (Y/N)'].map({'Y': 1, 'N': 0})

#Mentions
df['mentions_binary'] = df['Mentions? (Y/N)'].map({'Y': 1, 'N': 0})
df['video_binary'] = df['Video? (Y/N)'].map({'Y': 1, 'N': 0})


# Hashtag Count
df['Hashtags List'] = df['Hashtags List'].astype(str)
df['hashtag_count'] = df['Hashtags List'].apply(lambda x: len(x.split(',')) if isinstance(x, str) else 0)

#Number of Mentions
df['Mentions List'] = df['Mentions List'].astype(str)
df['hashtag_count'] = df['Mentions List'].apply(lambda x: len(x.split(',')) if x else 0)

#Width, Height
df = df[df['Dimensions'] != 'N/A']
df = df[df['Dimensions'].notna()]  # Drop NaN values
df[['width', 'height']] = df['Dimensions'].str.split('x', expand=True).astype(int)

#Area
df['area'] = df['width'] * df['height']

#Aspect Ratio
df['aspect_ratio'] = df['width'] / df['height']

# Step 1: Extract the hour from 'Time of Day of Post'
df['Hour'] = df['Time of Day of Post'].apply(lambda x: int(x.split(':')[0]))

# Step 2: One-hot encode the 'Hour' column
df_one_hot = pd.get_dummies(df['Hour'], prefix='Hour')

# Combine the one-hot encoded columns with the original DataFrame
df = pd.concat([df, df_one_hot], axis=1)

# Drop the original 'Time of Day of Post' and 'Hour' columns if not needed
df = df.drop(columns=['Time of Day of Post', 'Hour'])


# 6. Day of Week: One-hot encode
day_encoder = OneHotEncoder(sparse_output=False)
day_encoded = day_encoder.fit_transform(df[["Day of Week"]])
day_encoded_df = pd.DataFrame(day_encoded, columns=day_encoder.get_feature_names_out(["Day of Week"]))

# 7. Date of Post: Extract month, day, year
df[["month", "day", "year"]] = pd.to_datetime(df["Date of Post"]).apply(lambda x: [x.month, x.day, x.year]).to_list()


# 8. Engagement Rate, Likes, Comments: Normalize
scaler = StandardScaler()
df[["likes", "comments"]] = scaler.fit_transform(df[["Likes", "Comments"]])

# Combine all features into final DataFrame
final_df = pd.concat([df, day_encoded_df], axis=1)

print(final_df)

print(final_df.columns)

final_df = final_df[['Sentiment Polarity', 'video_binary', 
       'hashtag_binary', 'mentions_binary',
       'hashtag_count', 'width', 'height', 'area', 'aspect_ratio', 'Hour_4',
       'Hour_5', 'Hour_6', 'Hour_7', 'Hour_8', 'Hour_9', 'Hour_10', 'Hour_11',
       'Hour_12', 'Hour_13', 'Hour_14', 'Hour_15', 'Hour_16', 'Hour_17',
       'Hour_18', 'Hour_19', 'Hour_20', 'Hour_21', 'month', 'day', 'likes', 'Day of Week_Friday',
       'Day of Week_Monday', 'Day of Week_Saturday', 'Day of Week_Sunday',
       'Day of Week_Thursday', 'Day of Week_Tuesday', 'Day of Week_Wednesday','color_group_Black', 'color_group_Blue', 'color_group_Cyan',
       'color_group_Green', 'color_group_Magenta', 'color_group_Red',
       'color_group_White', 'color_group_Yellow']]

final_df = final_df.dropna()


print(final_df)



X = final_df.drop('likes', axis=1)
Y = final_df['likes']

# Split data into 60% training, 20% cross-validation, and 20% test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, Y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print("Training set size:", len(X_train))
print("Cross Validation set size:", len(X_val))
print("Test set size:", len(X_test))

# Creation of competing neural network models
first = X.shape[1]
print("Number of variables:", first)

# Define z-score threshold for success
z_score_threshold = 0  # Adjust this value as needed
y_train_class = (y_train > z_score_threshold).astype(int)
y_val_class = (y_val > z_score_threshold).astype(int)
y_test_class = (y_test > z_score_threshold).astype(int)

print("Class Distribution in Training Set:", y_train_class.value_counts())
print("Class Distribution in Validation Set:", y_val_class.value_counts())
print("Class Distribution in Test Set:", y_test_class.value_counts())

# Update the neural network for binary classification
def create_classification_model(units_list, input_shape=(X_train.shape[1],), activation='relu'):
    model = tensorflow.keras.models.Sequential()
    model.add(tensorflow.keras.layers.Dense(units_list[0], input_shape=input_shape, activation=activation))
    for units in units_list[1:]:
        model.add(tensorflow.keras.layers.Dense(units, activation=activation))
    model.add(tensorflow.keras.layers.Dense(1, activation='sigmoid'))  # Sigmoid activation for binary classification
    model.compile(
        optimizer='adam',  # Optimizer
        loss='binary_crossentropy',  # Binary classification loss
        metrics=['accuracy']  # Evaluate with accuracy
    )
    return model

# Define the structures to test
structures = [
    [64, 32],
    [64, 32, 16],
]

# Train and evaluate models for binary classification
results = []
for structure in structures:
    print(f"Training classification model with structure: {structure}")
    model = create_classification_model(structure)
    model.fit(X_train, y_train_class, epochs=10, batch_size=10, validation_data=(X_val, y_val_class), verbose=1)

    # Predict and evaluate on validation set
    y_val_pred_probs = model.predict(X_val).ravel()
    y_val_pred = (y_val_pred_probs > 0.5).astype(int)
    val_accuracy = accuracy_score(y_val_class, y_val_pred)

    # Predict and evaluate on test set
    y_test_pred_probs = model.predict(X_test).ravel()
    y_test_pred = (y_test_pred_probs > 0.5).astype(int)
    test_accuracy = accuracy_score(y_test_class, y_test_pred)

    results.append({
        'structure': structure,
        'val_accuracy': val_accuracy,
        'test_accuracy': test_accuracy,
        'model': model
    })

# Convert results to DataFrame for easy comparison
results_df = pd.DataFrame(results)
print(results_df)

# Select the best model based on validation accuracy
best_model_info = results_df.loc[results_df['val_accuracy'].idxmax()]
best_model_nn = best_model_info['model']
print(f"Best Neural Network Model: {best_model_info['structure']} layers")
print(f"Validation Accuracy: {best_model_info['val_accuracy']}")
print(f"Test Accuracy: {best_model_info['test_accuracy']}")

# Classification Report on Test Set
from sklearn.metrics import classification_report
print("Classification Report (Test Set):")
print(classification_report(y_test_class, y_test_pred))

# --- Decision Tree Logic + Random Forest Parameters ---
# Define the parameter grid
param_grid = {
    'n_estimators': [100],
    'criterion': ['entropy'],
    'max_depth': [None],
    'min_samples_split': [5],
    'min_samples_leaf': [1, 5],
    'max_features': ['log2']
}

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    verbose=2,
    n_jobs=-1
)

# Fit GridSearchCV
grid_search.fit(X_train, y_train_class)

# Get the best parameters and best score
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: ", grid_search.best_score_)

# Evaluate the best model on the validation set
best_rf = grid_search.best_estimator_

# Evaluate on training set
y_train_pred = best_rf.predict(X_train)
accuracy_train = accuracy_score(y_train_class, y_train_pred)
print(f"Train Accuracy: {accuracy_train}")
print("Train Confusion Matrix:\n", confusion_matrix(y_train_class, y_train_pred))
print("Train Classification Report:\n", classification_report(y_train_class, y_train_pred))

# Evaluate on validation set
y_val_pred = best_rf.predict(X_val)
accuracy_val = accuracy_score(y_val_class, y_val_pred)
print(f"Validation Accuracy with Best Model: {accuracy_val}")
print("Validation Confusion Matrix:\n", confusion_matrix(y_val_class, y_val_pred))
print("Validation Classification Report:\n", classification_report(y_val_class, y_val_pred))

# Evaluate on test set
y_test_pred = best_rf.predict(X_test)
accuracy_test_rf = accuracy_score(y_test_class, y_test_pred)
print(f"Test Accuracy with Best Model: {accuracy_test_rf}")
print("Test Confusion Matrix:\n", confusion_matrix(y_test_class, y_test_pred))
print("Test Classification Report:\n", classification_report(y_test_class, y_test_pred))

# Check for overfitting
if accuracy_train > accuracy_test_rf and accuracy_train - accuracy_test_rf > 0.05:
    print("The model is likely overfitting.")
else:
    print("The model does not seem to be overfitting.")

# Feature Importance Analysis
import matplotlib.pyplot as plt
importances = best_rf.feature_importances_
features = X_train.columns
plt.figure(figsize=(10, 6))
plt.barh(features, importances)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Random Forest Feature Importances')
plt.show()

# Final Model Selection
if accuracy_test_rf > accuracy_val:  # Compare Random Forest and NN validation accuracy
    final_model = best_rf
    print(f'Random Forest is the final model. Parameters: {grid_search.best_params_}. Test Accuracy: {accuracy_test_rf:.2f}.')
else:
    final_model = best_model_nn
    print(f'Neural Network is the final model. Test Accuracy: {accuracy_val:.2f}.')
