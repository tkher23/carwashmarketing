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
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error



# Example DataFrame
# Replace 'your_file.csv' with the path to your CSV file
file_path = 'quickquack_posts_data.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)



# 1. Text Vectorization for Post Caption
#tfidf = TfidfVectorizer(max_features=10)  # Adjust features as needed
#df_tfidf = tfidf.fit_transform(df["post_caption"]).toarray()
#df_tfidf = pd.DataFrame(df_tfidf, columns=tfidf.get_feature_names_out())

# 2. Sentiment Polarity: Already normalized (0-1)

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
       'Hour_18', 'Hour_19', 'Hour_20', 'Hour_21', 'Hour_22', 'Hour_23',
       'month', 'day', 'year', 'likes', 'comments', 'Day of Week_Friday',
       'Day of Week_Monday', 'Day of Week_Saturday', 'Day of Week_Sunday',
       'Day of Week_Thursday', 'Day of Week_Tuesday', 'Day of Week_Wednesday']]

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


def create_model(units_list, input_shape=(X_train.shape[1],), activation='relu'):
    model = tensorflow.keras.models.Sequential()
    model.add(tensorflow.keras.layers.Dense(units_list[0], input_shape=input_shape, activation=activation))
    for units in units_list[1:]:
        model.add(tensorflow.keras.layers.Dense(units, activation=activation))
    model.add(tensorflow.keras.layers.Dense(1, activation='linear'))  # Linear activation for regression output
    model.compile(
        optimizer='adam',  # Optimizer
        loss='mean_squared_error',  # Regression loss function
        metrics=[tensorflow.keras.metrics.RootMeanSquaredError()]  # RMSE metric
    )
    return model

# Define the structures to test
structures = [
    [64, 32],
    [64, 32, 16],
]

# Store metrics
results = []

# Train and evaluate models for each structure
for structure in structures:
    print(f"Training model with structure: {structure}")
    model = create_model(structure)
    model.fit(X_train, y_train, epochs=10, batch_size=10, verbose=1)  # Reduced epochs for quicker testing

    # Evaluate on validation set
    y_val_pred = model.predict(X_val).ravel()
    mse_val = mean_squared_error(y_val, y_val_pred)
    rmse_val = mse_val ** 0.5

    # Evaluate on training set
    y_train_pred = model.predict(X_train).ravel()
    mse_train = mean_squared_error(y_train, y_train_pred)
    rmse_train = mse_train ** 0.5

    # Evaluate on test set
    y_test_pred = model.predict(X_test).ravel()
    mse_test = mean_squared_error(y_test, y_test_pred)
    rmse_test = mse_test ** 0.5

    results.append({
        'structure': structure,
        'rmse_train': rmse_train,
        'rmse_val': rmse_val,
        'rmse_test': rmse_test,
        'model': model
    })

# Convert results to DataFrame for easy comparison
results_df = pd.DataFrame(results)
print(results_df)

# Select the best model based on validation RMSE
best_model_info = results_df.loc[results_df['rmse_val'].idxmin()]
best_model_nn = best_model_info['model']
print(f"Best Neural Network Model: {best_model_info['structure']} layers")
print(f"Train RMSE: {best_model_info['rmse_train']}")
print(f"Validation RMSE: {best_model_info['rmse_val']}")
print(f"Test RMSE: {best_model_info['rmse_test']}")
