import instaloader
import os
import json
import pandas as pd
from datetime import datetime
from PIL import Image
import numpy as np
import lzma
from textblob import TextBlob

def download_instagram_data(profile_name):
    # Initialize Instaloader
    loader = instaloader.Instaloader(
        download_videos=False,  # Include videos in the download
        download_video_thumbnails=False,  # Skip video thumbnails
        download_pictures=False,
        download_comments=False,
    )
    
    # Create a session (if login is needed)
    # Uncomment and replace with your credentials if required
    # loader.login("your_username", "your_password")

    # Download posts for the profile
    loader.download_profile(profile_name, profile_pic=False, fast_update=True)

# Function to initialize Instaloader and download metadata, images, and videos
def load_metadata_to_dataframe(profile_name):
    # Path to the directory containing JSON files (update the path as needed)
    posts_path = f'/Users/sohailmukadam/Desktop/Instragram Analyzer/{profile_name}'
    json_files = [file for file in os.listdir(posts_path) if file.endswith(".json") or file.endswith(".json.xz")]

    # List to store data for the DataFrame
    data = []

    if not json_files:
        print("No JSON or .json.xz files found.")
        return pd.DataFrame()  # Return an empty DataFrame

    for file in json_files:
        file_path = os.path.join(posts_path, file)
        try:
            # Check if the file is .json.xz and open it
            if file.endswith(".json.xz"):
                with lzma.open(file_path, 'rt', encoding='utf-8') as f:
                    post_data = json.load(f)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    post_data = json.load(f)

            # Extract data from the JSON structure
            node = post_data.get("node", {})

            # Caption and sentiment
            caption = node.get("caption", "")
            hashtags = [tag.strip("#") for tag in caption.split() if tag.startswith("#")]
            mentions = [mention.strip("@") for mention in caption.split() if mention.startswith("@")]
            hashtags_present = "Y" if hashtags else "N"
            mentions_present = "Y" if mentions else "N"
            sentiment = TextBlob(caption).sentiment.polarity if caption else 0

            # Timestamp and posting time
            timestamp = node.get("date", None)
            if timestamp:
                timestamp = datetime.fromtimestamp(timestamp)
                time_of_day = timestamp.strftime("%H:%M:%S")
                date_of_post = timestamp.strftime("%Y-%m-%d")
                day_of_week = timestamp.strftime("%A")
            else:
                time_of_day = "N/A"
                date_of_post = "N/A"
                day_of_week = "N/A"

            # Post details
            is_video = "Y" if node.get("is_video", False) else "N"
            dimensions = node.get("iphone_struct", {})
            dimensions_str = f"{dimensions.get('original_width', 'N/A')}x{dimensions.get('original_height', 'N/A')}"
            likes = node.get("edge_media_preview_like", {}).get("count", 0)
            comments = node.get("comments", 0)

            # Engagement rate
            followers = node.get("owner", {}).get("edge_followed_by", {}).get("count", 1)  # Follower count
            engagement_rate = ((likes + comments) / followers) * 100 if followers else 0

            # Append to data list
            data.append({
                "Post Caption": caption,
                "Sentiment Polarity": sentiment,
                "Hashtags? (Y/N)": hashtags_present,
                "Mentions? (Y/N)": mentions_present,
                "Hashtags List": ", ".join(hashtags),
                "Mentions List": ", ".join(mentions),
                "Time of Day of Post": time_of_day,
                "Day of Week": day_of_week,
                "Date of Post": date_of_post,
                "Video? (Y/N)": is_video,
                "Dimensions": dimensions_str,
                "Likes": likes,
                "Comments": comments,
                "Engagement Rate (%)": engagement_rate
            })

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    # Create a DataFrame from the collected data
    df = pd.DataFrame(data)
    return df

# Main function to execute the workflow
def main():
    profile_name = "mistercarwashhq"
    
    print("Step 1: Downloading Instagram data...")
    download_instagram_data(profile_name)
    
    print("Step 2: Loading metadata into DataFrame...")
    df = load_metadata_to_dataframe(profile_name)
    
    print("Step 3: Saving DataFrame to CSV...")
    df.to_csv(f"{profile_name}_posts_data.csv", index=False)
    print(f"Data saved to {profile_name}_posts_data.csv")

# Run the program
if __name__ == "__main__":
    main()
