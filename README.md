# 🤖 X-Bot Detection: Twitter/X Bot Account Classification Model

This project focuses on developing a **Machine Learning** model to classify accounts on Twitter/X as a **Bot** or **Human (Non-Bot)**, utilizing features derived from profile metadata and tweet behavior.

## 📊 Dataset

The dataset used is a structured and balanced subset of original Twitter/X data, encompassing user metadata and tweet details.

### Primary Data Columns

The final dataset used for modeling is formatted to combine profile and tweet features for each tweet entry:

| Data Column        | Description                                   | Feature Type |
| :----------------- | :-------------------------------------------- | :----------- |
| **User ID**        | Unique user identifier.                       | Profile      |
| **Tweet**          | Tweet text content.                           | Tweet        |
| **Retweet Count**  | The number of retweets for that tweet.        | Tweet        |
| **Mention Count**  | The number of mentions in the tweet.          | Tweet        |
| **Follower Count** | The number of user followers.                 | Profile      |
| **Verified**       | Verified account status (1/0).                | Profile      |
| **Bot Label**      | Target label: **1** for Bot, **0** for Human. | Target       |
| **Hashtag Count**  | The number of hashtags in the tweet.          | Tweet        |
| **URL Count**      | The number of URLs in the tweet.              | Tweet        |

## ⚙️ Data Preparation

The data preparation process was carried out extensively to handle the large data size and class imbalance:

1.  **Initial Stratified Sampling**: The original dataset consisted of **1,000,000 users**. A larger initial sample of **20,000 users** was taken while maintaining a bot:human ratio (14% bots).
2.  **Tweet Data Extraction**: For memory safety reasons, tweet data was extracted using a very small sample (2,000 users) from the large JSON file.
3.  **Final Merged Dataset**: The extracted and merged dataset contains **20,123 tweets** from **640 unique users**. The label distribution at this stage was highly imbalanced: **57 Bots (8.9%)** and **583 Humans (91.1%)**.
4.  **Data Balancing (Undersampling)**: To address _class imbalance_, **undersampling** was performed on the majority class (Non-Bot) based on users, resulting in a balanced dataset with a total of **3,900 rows** (1,950 Bot rows and 1,950 Non-Bot rows).

## 💡 Feature Engineering & Selection

New features were engineered to capture the behavioral characteristics of bots and humans:

| Engineered Feature              | Description                                                                                    |
| :------------------------------ | :--------------------------------------------------------------------------------------------- |
| **Time Interval Entropy (TIE)** | Measures the randomness or regularity of time intervals between tweets.                        |
| **Tweet Rate Per Day**          | The average rate of posting tweets per day.                                                    |
| **Follower Density**            | The number of followers divided by the account age (in seconds).                               |
| **Mean/Std Tweet Metrics**      | The Mean and Standard Deviation of `Retweet Count`, `Mention Count`, and `URL Count` per user. |

### Feature Selection (SelectKBest)

After feature engineering, **SelectKBest** was used to select the **8 most optimal features** for modeling:

- `Time_Interval_Entropy`
- `Tweet_Rate_Per_Day`
- `Follower_Count`
- `Follower_Density`
- `Mean_Mention_Count`
- `Std_Mention_Count`
- `Mean_URL_Count`
- `Verified`

## 🧠 Modeling & Evaluation

### Algorithm

The implemented classification model is **Support Vector Machine (SVM)**, which is known to be effective in high-dimensional feature spaces.

### Hyperparameter Tuning

Using **GridSearchCV** with 5-fold cross-validation, the optimal hyperparameters were found as:

- **Best Kernel:** `rbf`
- **Best Parameters:** `{'C': 1000, 'gamma': 0.001, 'kernel': 'rbf'}`

### Model Evaluation Results

Evaluation was performed on the balanced **testing data**:

| Metric                   | Score      |
| :----------------------- | :--------- |
| **Accuracy**             | 0.6522     |
| **Precision**            | 0.6364     |
| **Recall (Sensitivity)** | 0.6364     |
| **F1-Score**             | **0.6364** |

**Brief Interpretation:**
The SVM model achieved an **F1-Score of 0.6364**, demonstrating a moderate classification ability in distinguishing between bot and non-bot accounts on the balanced dataset. This result can be further improved by exploring other algorithms (such as Random Forest or Gradient Boosting) and more in-depth feature engineering.

## 📁 Project Structure

```bash
X-Bot-Detection/
├── data/
│   ├── README.md           # Original data column description.
│   ├── bot_detection_data.csv # Raw data before preparation.
│   └── x_dataset/          # Folder for original raw data (contains label.csv, edge.csv, user.json, tweet_0.json).
├── final_x_dataset/
│   ├── sampled_labels.csv  # List of selected users for the 20,000 sample.
│   ├── extracted_tweets.csv# Tweet data extracted from the small sample.
│   └── bot_detection_data_1.csv # Final merged dataset (20,123 tweets) before balancing.
├── preparation.ipynb       # Notebook for data preparation, sampling, extraction, merging, and data balancing.
└── x_model.ipynb           # Notebook for feature engineering, selection, SVM hyperparameter tuning, and model evaluation.
```
