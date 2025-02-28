import re
import random
from collections import Counter, defaultdict
import numpy as np 

def feature_length(ciphertext):
    """
    Returns the length of the ciphertext.

    :param ciphertext: The input ciphertext
    :type ciphertext: str
    :return: The length of the ciphertext
    :rtype: int
    """
    return len(ciphertext)


def feature_has_repeated_sequences(text, min_length=3):
    """
    Counts the number of repeated sequences of a given minimum length in the text.

    :param text: The input text to analyze for repeated sequences
    :type text: str
    :param min_length: The minimum length of sequences to consider, default is 3
    :type min_length: int
    :return: The count of sequences that are repeated more than once
    :rtype: int
    """

    if len(text) < min_length:
        return 0
    sequences = Counter(text[i:i + min_length] for i in range(len(text) - min_length + 1))
    repeated_count = sum(1 for count in sequences.values() if count > 1)
    return repeated_count

def feature_ic(text):
    """
    Calculates the Index of Coincidence (IC) for a given text.

    IC is a measure of how likely it is for two letters to be the same in a given text.
    It is calculated as the sum of the squares of the frequencies of each letter divided by the square of the length of the text minus one.

    :param text: The input text to calculate the IC for
    :type text: str
    :return: The Index of Coincidence for the given text
    :rtype: float
    """
    n = len(text)
    if n <= 1:
        return 0.0
    counts = Counter(text)
    numerator = sum(count * (count - 1) for count in counts.values())
    denominator = n * (n - 1)
    return numerator / denominator if denominator > 0 else 0.0

def average_ic_for_cosets(ciphertext):
    """
    Calculates the average Index of Coincidence (IC) for a given text over different coset sizes.

    The IC is calculated for each coset size from 3 to 25, and the average IC is returned as a dictionary,
    where the keys are the coset sizes and the values are the corresponding average ICs.

    :param ciphertext: The input ciphertext to calculate the average ICs for
    :type ciphertext: str
    :return: A dictionary of average ICs for each coset size
    :rtype: dict
    """

    avg_ic_by_m = {}
    
    for m in range(3, 26):
        cosets = [[] for _ in range(m)]
        for index, char in enumerate(ciphertext):
            coset_index = index % m
            cosets[coset_index].append(char)
        total_ic = 0.0
        valid_cosets = 0
        for coset in cosets:
            coset_str = ''.join(coset)
            ic = feature_ic(coset_str)
            total_ic += ic
            valid_cosets += 1
        
        if valid_cosets == 0:
            avg_ic = 0.0
        else:
            avg_ic = total_ic / valid_cosets
        avg_ic_by_m[m] = avg_ic
    
    return avg_ic_by_m


def feature_ic_english():
    """
    Returns the Index of Coincidence for the English language.

    The Index of Coincidence (IC) for English text is a constant value
    representing the probability that two randomly selected letters are the same.
    This function returns the IC value typically used for English, which is 0.066.

    :return: The Index of Coincidence for English
    :rtype: float
    """

    return 0.066


def calculate_twist_indices(text, max_m=25):
    """
    Calculates the Twist+ and Twist++ values for a given ciphertext.

    The Twist+ and Twist++ values are two metrics used to measure the
    periodicity of a given ciphertext. The Twist+ value is the average
    of the differences between the Index of Coincidence (IC) values of
    adjacent blocks of length m, normalized by m. The Twist++ value is
    the average of the differences between the Twist+ values of adjacent
    blocks of length m, normalized by m.

    :param str text: The input ciphertext to analyze
    :param int max_m: The maximum value of m to consider (default: 25)
    :return: A tuple containing the Twist+ and Twist++ values as lists
    :rtype: tuple[list[float], list[float]]
    """
    N = len(text)
    q = N // 12
    twist_indices = [0.0] * (max_m + 1)
    twist_plus = [0.0] * (max_m + 1)
    twist_plus_plus = [0.0] * (max_m + 1)
    def split_into_cosets(text, m):
        cosets = defaultdict(list)
        for idx, char in enumerate(text):
            coset_idx = idx % m
            cosets[coset_idx].append(char)
        return cosets
    
    def compute_coset_twist(coset):
        total = len(coset)
        if total == 0:
            return 0
        freq = defaultdict(int)
        for c in coset:
            freq[c] += 1
        freqs = sorted([v / total for v in freq.values()], reverse=True)
        freqs += [0.0] * (26 - len(freqs))
        twist = sum(freqs[13:]) - sum(freqs[:13])
        return twist
    
    for m in range(1, min(max_m, q) + 1):
        cosets = split_into_cosets(text, m)
        total_twist = sum(compute_coset_twist(coset) for coset in cosets.values())
        twist_indices[m] = (total_twist / m) * 100
        
    for m in range(2, min(max_m, q) + 1):
        avg_prev = np.mean([twist_indices[mu] for mu in range(1, m)] if m > 1 else [0])
        twist_plus[m] = twist_indices[m] - avg_prev
        
    for m in range(2, min(max_m, q)):
        if m + 1 <= q:
            avg_neighbors = (twist_indices[m-1] + twist_indices[m+1]) / 2
            twist_plus_plus[m] = twist_indices[m] - avg_neighbors
        else:
            twist_plus_plus[m] = 0.0 
            
    return twist_plus[2:], twist_plus_plus[2:]

def feature_hi7(ciphertext):
    """
    Calculates the Hi7 feature, which is the sum of the relative frequencies of the top 7 most common letters in the ciphertext.

    :param ciphertext: The input ciphertext to analyze
    :type ciphertext: str
    :return: The value of the Hi7 feature
    :rtype: float
    """
    freq = Counter(ciphertext)
    total = len(ciphertext)
    if total == 0:
        return 0
    rel_freqs = [count / total for _, count in freq.most_common(7)]
    return sum(rel_freqs)

def feature_delta7(ciphertext):
    """
    Calculates the Delta7 feature, which is the difference between the sum of the relative frequencies of the top 7 most common letters and the sum of the relative frequencies of the bottom 7 least common letters in the ciphertext.

    :param ciphertext: The input ciphertext to analyze
    :type ciphertext: str
    :return: The value of the Delta7 feature
    :rtype: float
    """
    freq = Counter(ciphertext)
    total = len(ciphertext)
    if total == 0:
        return 0
    sorted_freqs = sorted([count / total for count in freq.values()], reverse=True)
    top7 = sum(sorted_freqs[:7])
    bottom7 = sum(sorted_freqs[-7:]) if len(sorted_freqs) >= 7 else 0
    return top7 - bottom7

def extract_features(ciphertext):
    """
    Extracts a set of features from a given ciphertext.
    Features:
    - length: The length of the ciphertext
    - has_repeated_sequences: Whether the ciphertext contains repeated sequences
    - ic: The Index of Coincidence for the ciphertext
    - ic_english: The Index of Coincidence for the English language
    - twist_plus_2, twist_plus_3, ..., twist_plus_25: The Twist+ values for different block sizes
    - twist_plus_plus_2, twist_plus_plus_3, ..., twist_plus_plus_25: The Twist++ values for different block sizes
    - avg_ic_3, avg_ic_4, ..., avg_ic_25: The average IC values for different block sizes
    - hi7: The Hi7 feature
    - delta7: The Delta7 feature
    

    :param ciphertext: The ciphertext to extract features from
    :type ciphertext: str
    :return: A dictionary mapping feature names to values
    :rtype: dict
    """
   
    features = {"length": feature_length(ciphertext)}
    features["has_repeated_sequences"] = feature_has_repeated_sequences(ciphertext)
    features["ic"] = feature_ic(ciphertext)
    features["ic_english"] = feature_ic_english()

    twist_plus_values, twist_plus_plus_values = calculate_twist_indices(ciphertext)
    for m in range(len(twist_plus_values)):
        features[f'twist_plus_{m+2}'] = twist_plus_values[m]
    for m in range(len(twist_plus_plus_values)):
        features[f'twist_plus_plus_{m+2}'] = twist_plus_plus_values[m]
        
    avg_ic = average_ic_for_cosets(ciphertext)
    
    for m in range(3, 26):
        features[f'avg_ic_{m}'] = avg_ic[m]

    features["hi7"] = feature_hi7(ciphertext)
    features["delta7"] = feature_delta7(ciphertext)
    return features

