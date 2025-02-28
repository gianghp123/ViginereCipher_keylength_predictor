import re
import random
import os
import matplotlib.pyplot as plt


def read_txt_files(directory):
    text = ''
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
                text += content
    return text

def clean_text(text):
    """
    Cleans the given text by:
    - Removing numbers
    - Removing punctuation
    - Removing spaces
    - Keeping only lowercase English letters (a-z).
    
    Args:
        text (str): The input text to clean.
    
    Returns:
        str: The cleaned text with only lowercase English letters.
    """
    text = text.lower()

    cleaned_text = re.sub(r'[^a-z]', '', text)
    
    return cleaned_text


def split_text(text, min_len=200, max_len=500, samples_per_length=1300):
    """
    Splits text into non-overlapping chunks of specified lengths between min_len and max_len,
    ensuring an equal number of samples for each length.
    
    Args:
        text (str): The input text to split.
        min_len (int): Minimum chunk length (default: 200).
        max_len (int): Maximum chunk length (default: 500).
        samples_per_length (int): Number of samples per length (default: 1300, as in the paper).
    
    Returns:
        list: A list of text chunks with balanced samples across lengths.
    """
    chunks = []
    available_lengths = list(range(min_len, max_len + 1))
    total_samples_needed = samples_per_length * len(available_lengths)  # Tổng số mẫu cần (1,300 * 301)
    
    # Kiểm tra nếu văn bản đủ dài để tạo đủ số mẫu
    if len(text) < total_samples_needed * min_len:
        raise ValueError("Input text is too short to generate the required number of samples with minimum length.")

    index = 0
    length_counts = {length: 0 for length in available_lengths}  # Theo dõi số mẫu cho mỗi độ dài
    
    while index < len(text) and sum(length_counts.values()) < total_samples_needed:
        # Chọn độ dài tiếp theo sao cho đảm bảo số lượng mẫu ngang nhau
        remaining_samples = {length: samples_per_length - length_counts[length] for length in available_lengths}
        if all(count == samples_per_length for count in length_counts.values()):
            break  # Đã đủ số mẫu cho mọi độ dài
        
        # Lọc các độ dài còn lại cần mẫu
        valid_lengths = [length for length in available_lengths if remaining_samples[length] > 0]
        if not valid_lengths:
            break
        
        # Chọn ngẫu nhiên một độ dài từ các độ dài còn cần mẫu
        length = random.choice(valid_lengths)
        
        # Kiểm tra nếu còn đủ văn bản để tạo đoạn với độ dài này
        if index + length > len(text):
            # Nếu không đủ, thử các độ dài nhỏ hơn còn lại
            valid_shorter_lengths = [l for l in valid_lengths if index + l <= len(text)]
            if not valid_shorter_lengths:
                break  # Không còn đủ văn bản cho bất kỳ độ dài nào
            length = random.choice(valid_shorter_lengths)
        
        # Thêm đoạn văn bản vào danh sách
        chunks.append(text[index:index + length])
        length_counts[length] += 1
        index += length  # Di chuyển đến vị trí tiếp theo
    
    # Nếu không tạo đủ mẫu, cắt bớt hoặc cảnh báo
    if sum(length_counts.values()) < total_samples_needed:
        print(f"Warning: Only generated {sum(length_counts.values())} samples instead of {total_samples_needed}. "
              "Text may be too short or fragmented.")
    
    # Đảm bảo danh sách kết quả chỉ chứa các mẫu đã tạo
    return chunks

def plot_distribution(df, title="Feature Distribution", figsize=(16, 12)):
    """
    Plots a histogram of each feature in the given DataFrame.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the features to plot.
    title : str, optional
        The title of the plot. Defaults to "Feature Distribution".
    figsize : tuple, optional
        The size of the plot figure. Defaults to (16, 12).
    
    Returns
    -------
    None
    """
    df.hist(figsize=(16, 12), bins=30)
    plt.suptitle(title, fontsize=16)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()