import pandas as pd
import json
import time
from tqdm import tqdm
from polygon import RESTClient 
import os
from dotenv import load_dotenv

load_dotenv()
polygon_key = os.getenv("POLYGON_API_KEY")
client = RESTClient(polygon_key)

def is_price_above_cutoff(ticker: str, cutoff_price: float) -> bool:
    try:
        aggs = client.get_previous_close_agg(ticker, adjusted=True)
        latest_close = float(aggs[0].close)
        return latest_close >= cutoff_price
    except Exception as e:
        print(f"Dropping {ticker} - API Error: {e}")
        return False
    
def refine_master_index(input_json: str, output_json: str, min_days: int = 5, min_price: float = 15.00):
    print(f"Loading Master Ticker Index: {input_json}")
    
    with open(input_json, 'r') as f:
        master_dict = json.load(f)
        
    initial_count = len(master_dict)
    print(f"Total initial tickers: {initial_count:,}")
    
    # freq filter
    print(f"\n Dropping stocks mentioned on fewer than {min_days} unique days...")
    freq_filtered_dict = {
        ticker: dates 
        for ticker, dates in master_dict.items() 
        if len(dates) >= min_days
    }
    
    freq_survivors = len(freq_filtered_dict)
    print(f"Dropped {initial_count - freq_survivors:,} obscure tickers.")
    print(f"Remaining: {freq_survivors:,}")
    
    # stock filter
    print(f"\n Dropping stocks under ${min_price:.2f})...")
    
    final_clean_dict = {}
    
    # use tqdm for progress bar for the API calls
    for ticker, dates in tqdm(freq_filtered_dict.items(), desc="Checking Stock Prices"):
        
        if is_price_above_cutoff(ticker, min_price):
            final_clean_dict[ticker] = dates
            
    # save
    with open(output_json, 'w') as f:
        json.dump(final_clean_dict, f, indent=4)
        
    print(f"\n saved to {output_json}")
    print(f"usable stocks for ML Training: {len(final_clean_dict):,}")

if __name__ == "__main__":
    INPUT_MAPPING = "ticker_date_mapping.json"
    OUTPUT_MAPPING = "final_tickers_date_filtered.json"
        
    MIN_DAYS_MENTIONED = 5
    PRICE_CUTOFF = 15.00 
        
    if os.path.exists(INPUT_MAPPING):
        refine_master_index(INPUT_MAPPING, OUTPUT_MAPPING, MIN_DAYS_MENTIONED, PRICE_CUTOFF)
    else:
        print(f"Cannot find {INPUT_MAPPING}")

