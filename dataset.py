import pandas as pd
import tqdm
from extract_features import extract_features
from utils import clean_text, split_text
from viginere import encrypt_vigenere, get_random_keyword
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import joblib
import os 

def create_dataset(text: str):
    """
    Creates a dataset of Vigen re encrypted texts with their corresponding features and key lengths.

    Parameters
    ----------
    text : str
        The input text to encrypt and extract features from.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the extracted features and key lengths of the input text.
    """
    data = []
    cleaned_text = clean_text(text)
    samples_per_length = (len(cleaned_text) / 350) // 301
    print(samples_per_length)
    chunks = split_text(cleaned_text, samples_per_length=samples_per_length)
    for chunk in tqdm.tqdm(chunks):
        key = get_random_keyword()
        encrypted_text = encrypt_vigenere(chunk, key)
        features = extract_features(encrypted_text)
        features["key_length"] = len(key)
        data.append(features)

    # Convert to DataFrame
    df = pd.DataFrame(data)
    return df


def rescale_dataset(dataset, save_path_standard_scaler='standard_scaler_twist.pkl', save_path_minmax_scaler='minmax_scaler_other.pkl'):
    """
    Normalize specific columns of the dataset using appropriate normalization techniques, with the ability to save and load scalers from a file.

    Parameters
    -------
    dataset : pd.DataFrame
        The input DataFrame containing the columns to be normalized along with other features.

    save_path_standard_scaler : str
        Path to save/load the StandardScaler for twist_columns.

    save_path_minmax_scaler : str
        Path to save/load the MinMaxScaler for other_columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'twist' columns normalized using StandardScaler,
        'length' and 'has_repeated_sequences' columns normalized using MinMaxScaler,
        while other specified columns remain unchanged.
    """

    # Xác định các cột cần chuẩn hóa
    twist_columns = [col for col in dataset.columns if 'twist' in col]
    other_columns = ['length', 'has_repeated_sequences']
    normal_columns = ['key_length', 'ic_english', 'ic', 'hi7', 'delta7'] + [col for col in dataset.columns if 'avg_ic' in col]
   
    # Đặt index của dataset làm index mặc định
    original_index = dataset.index

    # Xử lý StandardScaler cho twist_columns
    if os.path.exists(os.path.join('scalers', save_path_standard_scaler)):
        print(f"Tải StandardScaler cho twist_columns từ {os.path.join('scalers', save_path_standard_scaler)}")
        standard_scaler_twist = joblib.load(os.path.join('scalers', save_path_standard_scaler))
        scaled_twist_df = pd.DataFrame(standard_scaler_twist.transform(dataset[twist_columns]), 
                                     columns=twist_columns, index=original_index)
    else:
        print(f"Khởi tạo và fit StandardScaler cho twist_columns, sau đó lưu vào {os.path.join('scalers', save_path_standard_scaler)}")
        standard_scaler_twist = StandardScaler()
        scaled_twist_df = pd.DataFrame(standard_scaler_twist.fit_transform(dataset[twist_columns]), 
                                     columns=twist_columns, index=original_index)
        joblib.dump(standard_scaler_twist, os.path.join('scalers', save_path_standard_scaler))

    # Xử lý MinMaxScaler cho other_columns
    if os.path.exists(os.path.join('scalers', save_path_minmax_scaler)):
        print(f"Tải MinMaxScaler cho other_columns từ {os.path.join('scalers', save_path_minmax_scaler)}")
        minmax_scaler_other = joblib.load(os.path.join('scalers', save_path_minmax_scaler))
        scaled_other_df = pd.DataFrame(minmax_scaler_other.transform(dataset[other_columns]), 
                                     columns=other_columns, index=original_index)
    else:
        print(f"Khởi tạo và fit MinMaxScaler cho other_columns, sau đó lưu vào {os.path.join('scalers', save_path_minmax_scaler)}")
        minmax_scaler_other = MinMaxScaler()
        scaled_other_df = pd.DataFrame(minmax_scaler_other.fit_transform(dataset[other_columns]), 
                                     columns=other_columns, index=original_index)
        joblib.dump(minmax_scaler_other, os.path.join('scalers', save_path_minmax_scaler))

    # Kết hợp các cột đã chuẩn hóa và cột giữ nguyên, đảm bảo index khớp
    scaled_df = pd.concat([scaled_twist_df, scaled_other_df], axis=1)
    scaled_df[normal_columns] = dataset[normal_columns]  # Gán trực tiếp với index khớp

    return scaled_df

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.features = self.data.drop(columns=["key_length"]).values.astype("float32")
        self.labels = self.data["key_length"].values.astype("int64")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        label = self.labels[idx] - 3
        return torch.tensor(self.features[idx]), torch.tensor(label)