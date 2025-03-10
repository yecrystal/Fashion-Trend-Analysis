import os
import pandas as pd
import numpy as np
import emot 

def read_data(path: str) -> pd.DataFrame:
    """
    Read data from a json file and return a pandas DataFrame
    
    Args: 
    path(str): Path to the json file
    
    Returns: 
    pd.DataFrame: A pandas dataFrame of the json file
    """

    df = pd.read_json(path)
    return df

def replace_emojis_with_text(text:str) -> str:
    """
    Replace emojis with text
    
    Args:
    text(str): Text with emojis
    
    Returns:
    str: Text with emojis replaced by textual descriptions
    """
    
    emot_obj = emot.emot()

    try:
        emoji_info = emot_obj.emoji(text)
        num_emojis = len(emoji_info['value'])

        for i in range(num_emojis):
            text = text.replace(emoji_info['value'][i], emoji_info['mean'][i])
    except Exception as e:
        print(f'An error occurred while processing the text: {text}. The error is as follows: {e}')

    return text

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the data in the DataFrame and return the processed DataFrame
    
    Args:
    df(pd.DataFrame): A pandas DataFrame
    
    Returns:
    pd.DataFrame: A pandas DataFrame with the processed data
    """

    # Selecting the required columns
    df = df[['id', 'type', 'commentsCount', 'likesCount', 'latestComments', 'images']]

    # Renaming the columns
    new_columns = {'id': 'id', 'commentsCount': 'n_comments', 'likesCount': 'n_likes', 'latestComments': 'comments', 'images': 'image'}
    df = df.rename(columns=new_columns)

    # Filtering out rows where type is "Video", likes count is -1.0 and image is not null
    df = df[(df["type"] != "Video")] 
    df = df[df["n_likes"] != -1.0]
    df = df[df["image"].notna()]

    # Removing rows with no image
    df = df[df["image"].apply(len) > 0]
    
    # Selecting the first image if there are multiple images
    df["image"] = df["image"].apply(lambda x: x[0])
    
    # Extracting text from comments
    df["comments"] = df["comments"].apply(lambda x: [i["text"] for i in x if "text" in i])
    
    # Resetting the id
    df.reset_index(drop=True, inplace=True)
    df["id"] = df.index + 1
    
    # Converting empty lists in comments to np.nan and creating a separate dataframe for comments
    df["comments"] = df["comments"].apply(lambda x: x if isinstance(x, list) and x else np.nan)
    df_comments = df.explode("comments")[["id", "comments"]]
    
    # Replace emojis in comments with their text descriptions
    df_comments["comments"] = df_comments["comments"].apply(replace_emojis_with_text)
    
    # Removing comments column from the original dataframe
    df = df.drop("comments", axis=1)
    
    return df, df_comments

def save_data(df: pd.DataFrame, df_comments: pd.DataFrame) -> None:
    """
    Save the DataFrame and comments dataframe to a csv file
    
    Args:
    df(pd.DataFrame): Dataframe to be saved
    df_comments(pd.DataFrame): Comments dataframe to be saved
    
    Returns:
    None
    """
    
    df.to_csv("data/processed_data.csv", index=False)
    df_comments.to_csv("data/processed_comments.csv", index=False, sep=';')

def main():
    path = "data/posts_1.json"
    path2 = "data/posts_2.json"

    # Check if posts_1.json exists, if not, use posts_2.json
    if os.path.exists(path):
        df_1 = read_data(path)
    else:
        print(f"File {path} not found, proceeding with {path2}")
        df_1 = pd.DataFrame()  # Create an empty dataframe

    df_2 = read_data(path2)
    df = pd.concat([df_1, df_2])
    df, df_comments = process_data(df)
    save_data(df, df_comments)
    
    
if __name__ == "__main__":
    main()
