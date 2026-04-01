# WallStreetBets Sentiment Analysis & Market Regime Detection
<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas" />
  <img src="https://img.shields.io/badge/Apache_Parquet-E25A1C?style=for-the-badge&logo=apache&logoColor=white" alt="PyArrow/Parquet" />
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter Notebook" />
</div>
<br>
**A Machine Learning pipeline mapping raw Reddit natural language data to End-Of-Day stock returns, featuring hyperparameter tuning and Concept Drift analysis.**

## 📖 Executive Summary
This project aims to predict daily stock market movements by analyzing the sentiment and specific vocabulary of the r/WallStreetBets subreddit. By engineering a custom Natural Language Processing (NLP) pipeline and mapping it to historical End-Of-Day (EOD) price action, this project tests the viability of social media sentiment as a leading indicator for financial markets. 

Crucially, this pipeline successfully demonstrates **Concept Drift (Market Regime Change)**. While the model achieves a strong mathematical edge during specific historical periods (e.g., the 2021 Meme Stock era), out-of-sample testing proves that static "Bag of Words" models decay as social media slang naturally evolves. 

## 🏗️ Architecture & Tech Stack

### Data Ingestion & Engineering
* **Text Data:** Processed massive `.zst` torrent archive dumps of historical Reddit submissions to bypass API rate limits and secure a complete historical dataset.
* **Price Data:** Fetched historical daily price action using the **EODHD API** to establish binary classification labels (Up Day = 1, Down Day = 0).
* **Storage Optimization:** Utilized **Pandas** and **PyArrow** to merge and compress millions of data points into highly optimized `.parquet` files, drastically reducing memory overhead and read/write times.

### NLP & Machine Learning Pipeline
* **Custom Vectorization:** Built a `TfidfVectorizer` pipeline featuring domain-specific financial stop-words (filtering out calendar/date noise) to isolate pure financial sentiment.
* **Algorithm Testing:** Benchmarked Multinomial Naive Bayes (with Laplace smoothing to prevent zero-frequency crashes) against Logistic Regression and Random Forest architectures.
<img width="421" height="338" alt="Screenshot 2026-03-31 at 7 54 48 PM" src="https://github.com/user-attachments/assets/8e55900b-9c8d-4770-8017-4f885da8e5ea" />

  
* **Hyperparameter Tuning:** Deployed a multi-threaded `GridSearchCV` matrix targeting the `F1-Score` to find the mathematically optimal vocabulary limits (`max_features`), n-gram ranges, and decision tree depths.

## 📊 Results & The "Concept Drift" Discovery

The overnight Grid Search optimizer trained 162 total fits using 3-Fold Cross Validation. 

During the Cross-Validation phase on the 2012–2022 training data, the Random Forest model achieved **In-Sample F1-Scores exceeding 61%**, proving that a distinct, tradable mathematical edge exists within the Reddit text. 

However, when forced to trade blindly on the **2023–2025 Out-Of-Sample Test Set**, the performance reverted to a ~50% baseline. 
<img width="421" height="338" alt="Screenshot 2026-03-31 at 7 54 48 PM" src="https://github.com/user-attachments/assets/80bd07c3-7b4e-4651-8c08-269f5aff6ea5" />

### The Takeaway
The algorithm perfectly learned the slang of the 2021 Meme Stock era (prioritizing words like *robinhood*, *gme*, and *squeeze*). When the market regime shifted in 2023 to focus on the AI supercycle, the static NLP model suffered from Concept Drift. This mathematically proves that a successful social sentiment strategy requires **Continuous Walk-Forward Retraining** to dynamically adapt to evolving internet culture. In all honesty, the Walk-Forward method may not be able to find alpha as well due to simply the noise of unstructured data. Additionally, due to the fact that I used binary labels, the models are in danger of falling into the trap of Accuracy vs Expectancy. For the Naive Bayes and Random Forest models, a -0.01% day and a -50% crash is mathematically identical for the same ticker. So even if these models have a high accuracy of 70%, a portfolio that it handles could be crushed by massive 20% drawdowns that were not predicted. A more refined approach is to instead use the absolute percentage of the returns in training in place of the binary labels. However that would mean training the models to predict the exact percentage, which is extremely hard to accomplish. So a better alternative may be to incorporate an industry standard of using "baskets" or Multi-Class Categorization: "2" would represent a return 
(> +5%), "1" is anything from (+0.5% to +5%), "0" is just sideways movements that can be counted as noise (-0.5% to +0.5%), "-1" is a downday of anywhere between (-5% to -0.5%), and "-2" is a really bad crash (< -5%). 

However I am still skeptical about this way of finding alpha. Perhaps ML models are best used as supervisors to already viable strategies on their own, where instead of trying to predict the market, the model monitors if strategy is continuously profitable. If not, the parameters of the strategy is reconsidered or at the worst case instantly liquidate the portfolio or buy put options.

## 🚀 How to Reproduce & Data Pipeline

This project was built from the ground up using raw, historical internet archives rather than pre-cleaned datasets. To fully reproduce this pipeline, follow these data engineering steps:

### 1. Reddit Data Acquisition (Torrent)
The core natural language data was sourced from massive historical Reddit submission dumps (commonly hosted via Academic Torrents or Web Archive). 
* Download the desired monthly/yearly Reddit submission archives. These are highly compressed as `.zst` (Zstandard) files.
* I personally used the Reddit comments/submissions 2005-06 to 2025-12
* **Do not attempt to unzip these directly into RAM.** A single month of raw Reddit data can exceed 50GB uncompressed. 

### 2. Stream-Processing & Text Extraction
To isolate the relevant financial data without crashing local memory:
* Use a stream-processing script (via the `zstandard` Python library or CLI tools) to read the archive line-by-line.
* Parse each line as a JSON object.
* Filter and retain only the rows where the `"subreddit"` field strictly matches `"wallstreetbets"`. 
* Extract the `"title"`, `"selftext"`, `"flair"`, `"upvote_ratio"` and `"created_utc"` fields, and parse out any mentioned stock tickers (cashtags).

### 3. Bulk Ticker Validation (The Massive API)
Because r/WallStreetBets frequently uses fake or meme tickers (e.g., `$ROPE`, `$WENDYS`), the extracted cashtags required rigorous cleaning before price mapping.
* The extracted tickers were passed through the **Massive API** to verify if they represented actual, tradable assets. 
* This specific API was chosen for its **unlimited call rates**, allowing the pipeline to aggressively bulk-verify hundreds of thousands of unique text strings without being throttled. 
* The cleaned, validated dataset was then saved locally as a highly compressed `.parquet` file.

### 4. Financial Price Mapping
* Utilize the **EODHD API** to pull historical End-Of-Day price action for the validated stock tickers.
* Calculate the daily percentage change, and map this to a binary classification label (`1` for an upward trend, `0` for a downward trend).
* Save this mapping as `market_labels.csv`.

### 5. Running the Machine Learning Pipeline
Once the `.parquet` text data and `.csv` market labels are locally available in your directory:
1. Run all the training functions in the machine learning pipeline file
2. When getting to the Random Forest Training and the Grid Search function, beware that it is very computationally expensive; adjust `n_jobs` in the script based on your CPU/RAM constraints.*
3. Perhaps try to implment a Walk-Forward method and see if that detects more of an edge than the static and 3- Fold Cross validation model
