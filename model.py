import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer

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

final_df = final_df[['Sentiment Polarity','Date of Post', 'Video? (Y/N)', 'Dimensions', 'Likes', 
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





