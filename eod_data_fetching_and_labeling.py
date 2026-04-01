import pandas as pd
import os
import json
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from eodhd import APIClient
from tqdm import tqdm

load_dotenv()
api = APIClient(os.getenv("EODHD_API_KEY"))

def fetch_data_eodhd_from_day_T(ticker: str, T: str, window: int): # T is date, written as "YYYY-MM-DD"
    start_date = datetime.strptime(T, "%Y-%m-%d") 
    buffer_days = int(window * 1.5) + 5 
    end_date = start_date + timedelta(days=buffer_days)
    to_date_str = end_date.strftime("%Y-%m-%d")
    eod_data = api.get_eod_historical_stock_market_data(symbol=ticker, 
                                                        period='d', 
                                                        from_date = T, 
                                                        to_date = to_date_str, 
                                                        order='a'
                                                        )
    df = pd.DataFrame(eod_data)

    if df.empty:
        return df
        
    df.columns = df.columns.str.strip()
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date']).dt.date
    df.set_index('date', inplace=True)

    return df.head(window)

def calculate_returns_during_window_from_postdate_and_ticker(ticker: str, post_date: str, window: int):
    try:
        total_days_needed = window + 1
        eod_df_after_post = fetch_data_eodhd_from_day_T(ticker=ticker, T=post_date, window=total_days_needed)
        # if invalid data
        if len(eod_df_after_post) < window:
            return None 

        entry_price = eod_df_after_post.iloc[1]['open'] # to sim live entering next day
        exit_price = eod_df_after_post.iloc[-1]['close']
        percent_return = (exit_price - entry_price) / entry_price
        return percent_return
    
    except Exception as e:
        print(f"Error calculating return for {ticker} on {post_date}: {e}")
        raise e

def label_calculated_returns(return_as_percent: float):
    if return_as_percent is None:
        return -1 #invalid data
    threshold = 0.0  
    if return_as_percent > threshold:
        return 1  
    else:
        return 0


def process_api_batch(ticker_dict: dict, existing_records: set, window: int, daily_limit: int):
    """
    Handles the actual API iteration, rate limiting, and progress tracking separately.
    """
    new_labels = []
    api_calls_today = 0
    limit_reached = False
    
    total_required_calls = sum(len(dates) for dates in ticker_dict.values())
    calls_remaining = total_required_calls - len(existing_records)

    print(f"calls left for today:{calls_remaining:,} network requests left to process.")
    
    # the progress bar
    with tqdm(total=min(calls_remaining, daily_limit), desc="Generating Labels") as pbar:
        
        for ticker, dates in ticker_dict.items():
            if limit_reached:
                break
                
            for date_str in dates:
                # CHECKPOINT: Skip if we already did this one yesterday!
                if (ticker, date_str) in existing_records:
                    continue
                    
                if api_calls_today >= daily_limit:
                    print("\nDAILY API LIMIT REACHED! saving progress")
                    limit_reached = True
                    break
                    
                # return percentage
                raw_return = calculate_returns_during_window_from_postdate_and_ticker(ticker, date_str, window)
                api_calls_today += 1
                
                # ml labelling (1, 0, or -1)
                ml_label = label_calculated_returns(raw_return)
                
                new_labels.append({
                    'tickers': ticker,
                    'date_str': date_str,
                    'label': ml_label
                })
                
                # update progress bar
                pbar.update(1)
                time.sleep(0.05) 
    return new_labels, limit_reached


def build_market_labels(json_mapping: str, output_csv: str, window: int = 5):
    """
    Loads mapping, checks progress, delegates API calls to process_api_batch, and saves results.
    """
    print(f"Loading ticker mapping from {json_mapping}...")
    with open(json_mapping, 'r') as f:
        ticker_dict = json.load(f)
        
    master_labels = []
    existing_records = set()
    
    # --- CHECKPOINT LOGIC: Load existing progress ---
    if os.path.exists(output_csv):
        print(f"Found existing {output_csv}. Loading previous progress...")
        df_existing = pd.read_csv(output_csv)
        master_labels = df_existing.to_dict('records')
        existing_records = set(zip(df_existing['tickers'], df_existing['date_str']))
        print(f"Loaded {len(existing_records):,} previously completed labels.")
        
    # check if done
    total_required_calls = sum(len(dates) for dates in ticker_dict.values())
    if total_required_calls - len(existing_records) <= 0:
        print("All labels have already been fetched! You are ready for ML.")
        return

    DAILY_API_LIMIT = 99900 
    new_labels, limit_reached = process_api_batch(
        ticker_dict=ticker_dict, 
        existing_records=existing_records, 
        window=window, 
        daily_limit=DAILY_API_LIMIT
    )
    
    master_labels.extend(new_labels)
            
    print("\nCleaning up missing/invalid data...")
    df_labels = pd.DataFrame(master_labels)
    
    # Drop any rows where the API failed or the stock wasn't trading (-1)
    initial_len = len(df_labels)
    df_labels = df_labels[df_labels['label'] != -1]
    
    print(f"Dropped {initial_len - len(df_labels):,} invalid API responses.")
    
    print(f"Saving {len(df_labels):,} perfect financial labels to {output_csv}...")
    df_labels.to_csv(output_csv, index=False)
    print("complete, data is safely saved")
    
    if limit_reached:
        print("NOTE: daily API limit reached. Run this exact script again tomorrow to finish")


if __name__ == "__main__":
    INPUT_JSON = "final_tickers_date_filtered.json"
    
    OUTPUT_CSV = "market_labels.csv"
    
    PREDICTION_WINDOW = 2
    
    if os.path.exists(INPUT_JSON):
        build_market_labels(INPUT_JSON, OUTPUT_CSV, PREDICTION_WINDOW)
    else:
        print(f"Error: Cannot find {INPUT_JSON}.")