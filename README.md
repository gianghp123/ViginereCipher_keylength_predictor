
# Predicting Vigenère Cipher Key Length Using Deep Learning

## 1. Introduction
This project is inspired by the research paper *"An Artificial Neural Network Approach to Finding the Key Length of the Vigenère Cipher"* by Christian Millichap and Yeeka Yau. The goal is to implement a Neural Network utilizing the most effective features extracted from encrypted text to predict the key length of the Vigenère cipher.

## 2. Summary of the Paper
The paper combines traditional cryptanalysis techniques such as the **Index of Coincidence (IC)** with modern algorithms like **twist-based algorithms** to determine the key length. This hybrid approach of traditional and machine learning-based techniques enhances the efficiency of cryptanalysis.

## 3. Key Features (Based on Table 4)
### Important Features Used:
- **Length of ciphertext (N)**: The length of the encrypted text.
- **Has repeated sequences**: Indicates the presence of repeated trigrams or quadgrams.
- **Index of Coincidence (IC) of the ciphertext**:
  \[ IC(M) = \frac{\sum_{i=1}^{26} f_i (f_i - 1)}{N(N-1)} \]
  where \( f_i \) is the frequency of letter \( i \).
- **Index of Coincidence of English text**: Fixed at 0.066.
- **Twist⁺ indices \( T^+(M,m) \)** for \( 2 \leq m \leq 25 \)
- **Twist⁺⁺ indices \( T^{++}(M,m) \)** for \( 2 \leq m \leq 25 \)
- **Average IC for cosets \( m \) where \( 3 \leq m \leq 25 \)**
- **Hi-7**: Sum of the percentage frequency of the 7 most common letters.
- **Δ-7**: Difference between the 7 most and 7 least common letters.

## 4. Model Implementation
The scaled dataset is published at [!dataset](https://huggingface.co/datasets/gianghp/ViginereCipher_features)
### 4.1 Data Preparation
Training data was collected from multiple text files (Kaggle) and preprocessed:
- Removing numbers, punctuation, and extra whitespace while converting all characters to lowercase.
- Splitting into non-repeating text segments of length **200–500 characters**.
- Encrypting with the **Vigenère cipher**, using random key lengths between **3 and 25**.
#### Training Dataset: 287,319 samples
#### Validation Dataset: 50,704 samples
#### Test Dataset Distribution:
| Text Length | Number of Samples |
|-------------|------------------|
| 200 - 299   | 4,800            |
| 300 - 399   | 4,800            |
| 400 - 500   | 4,800            |

### 4.2 Feature Extraction
- Features were computed using Python based on **9 feature groups** from Table 4.
- A total of **77 feature values** were computed.
- Feature scaling:
  - Large positive values (e.g., ciphertext length) scaled to **(0,1)**.
  - Twist⁺ and Twist⁺⁺ values scaled to **(-2.5, 2.5)**.

### 4.3 Model Architectures

#### 4.3.1 Feedforward Neural Network (FNN) as the paper
Implemented using **PyTorch** with the following structure:
- **Input Layer:**
  - **Dataset A**: 54 features (24 Twist⁺, 24 Twist⁺⁺, 6 other key features)
  - **Dataset B**: 77 features (adds 23 Average IC values for cosets)
- **Hidden Layers:**
  - 3 fully connected layers: **256 → 128 → 23 neurons**.
  - **LeakyReLU(0.01)** activation.
  - **BatchNorm + Dropout (0.3)** to prevent overfitting.
- **Output Layer:**
  - **23 neurons** (corresponding to key lengths 3-25).
  - No activation (Softmax integrated into loss function).
- **Optimizer:** AdamW (learning rate = 0.001, weight decay = 0.01).
- **Loss Function:** Categorical Cross-Entropy.
- **Learning Rate Scheduler:** ReduceLROnPlateau (reduce LR by 0.5 after 3 epochs with no improvement).

#### 4.3.2 Convolutional Neural Network (CNN)
Designed for sequence-based feature learning. Compared to FNN, it uses **convolutional layers** to capture dependencies between sequential features.

- **Input Layer:** Same as FNN.
- **Convolutional Layers:**
  - **2 Conv1D layers** (32 and 64 filters).
  - **LeakyReLU(0.01) activation**.
  - **BatchNorm1d for stability**.
- **Fully Connected Layers:**
  - **256 → 128 → 23 neurons**.
  - **LeakyReLU(0.01)**, **BatchNorm1d**, **Dropout(0.3)**.
- **Output Layer:**
  - **23 neurons** (corresponding to key lengths 3-25).
  - No activation (Softmax integrated into loss function).
- **Loss Function:** Categorical Cross-Entropy.
- **Optimizer:** AdamW (learning rate = 0.001, weight decay = 0.01).
- **Learning Rate Scheduler:** ReduceLROnPlateau.

### Comparison: FNN vs. CNN
- **CNN improved accuracy by ~8% compared to FNN**.
- CNN **better captured sequential dependencies** in features like **twist+ indices**.
- **Only CNN results are presented**, as it achieved the highest accuracy.

## 5. Results & Evaluation
- The final CNN model achieved **98.13% accuracy** on validation.
- **Significant accuracy improvements** were observed due to convolutional layers learning sequential dependencies in features, effectively capturing relationships between twist⁺ and twist⁺⁺ indices.
- Compared to the results reported in the referenced paper, which achieved **an accuracy of 89.2% on their best model**, the CNN implementation significantly outperformed it, achieving a **9.0% higher validation accuracy**. Additionally, our model demonstrated improved generalization when handling shorter ciphertexts, maintaining strong predictive performance across varying text lengths.
- The model struggled slightly with very short ciphertexts (200-299 characters), where accuracy dropped by around **3% compared to longer ciphertexts**, suggesting a need for additional feature engineering or larger training samples for short sequences.
