# preprocess_to_tfrecord.py
# VERSION 3.2: Added Example Counting for Fast Startup
#
# FIXES & IMPROVEMENTS:
#   1. Added Example Counting: The script now counts train, validation, and test examples
#      during generation and saves these counts to `metadata.pkl`. This allows the training
#      script to start instantly without needing to count records manually.
#   2. Complete Vocabulary (Memory-Safe): Scans both clicks and buys files
#      in a memory-efficient way to build a truly global vocabulary.
#   3. Robust Session Carry-over: Logic for handling sessions that span
#      across chunks is correctly based on the post-merge dataframe.
#   4. Timezone Correction: All timestamps are standardized to be timezone-naive
#      (based on UTC) to prevent comparison errors.

import os
import gc
import logging
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
class Config:
    DATA_PATH = '/Users/namshen/ACMSA25/DATA/Transformer/yoochoose-data'
    CLICKS_FILE = 'yoochoose-clicks.dat'
    BUYS_FILE = 'yoochoose-buys.dat'
    OUTPUT_PATH = 'processed_tfrecord/'
    MAX_SEQUENCE_LENGTH = 50
    MIN_SEQUENCE_LENGTH = 3
    CHUNK_SIZE = 5_000_000

# --- TFRecord Helper Functions ---
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def create_tf_example(sequence: dict) -> tf.train.Example:
    feature = {
        'items': _int64_feature(sequence['items']),
        'categories': _int64_feature(sequence['categories']),
        'time_deltas': _float_feature(sequence['time_deltas']),
        'label': _int64_feature([sequence['label']]),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def fit_encoders_and_get_splits_memory_safe(config: Config) -> tuple:
    logger.info("Scanning all data in chunks to build complete vocabulary and find time range...")
    
    clicks_path = os.path.join(config.DATA_PATH, config.CLICKS_FILE)
    buys_path = os.path.join(config.DATA_PATH, config.BUYS_FILE)
    
    item_ids = {0}
    categories = {'-1'}
    min_timestamp, max_timestamp = None, None

    # Scan clicks file
    with pd.read_csv(clicks_path, header=None, names=['session_id', 'timestamp', 'item_id', 'category'],
                     dtype={'category': str}, chunksize=config.CHUNK_SIZE) as reader:
        for chunk in tqdm(reader, desc="Scanning Clicks File"):
            chunk_timestamps = pd.to_datetime(chunk['timestamp'], errors='coerce', utc=True).dt.tz_localize(None)
            
            item_ids.update(chunk['item_id'].dropna().astype(int).unique())
            categories.update(chunk['category'].dropna().unique())
            
            if not chunk_timestamps.dropna().empty:
                current_min, current_max = chunk_timestamps.min(), chunk_timestamps.max()
                if min_timestamp is None or current_min < min_timestamp: min_timestamp = current_min
                if max_timestamp is None or current_max > max_timestamp: max_timestamp = current_max

    # Scan buys file
    logger.info("Scanning Buys File for any new items...")
    buys_df = pd.read_csv(buys_path, header=None, names=['session_id', 'timestamp', 'item_id', 'price', 'quantity'])
    buys_timestamps = pd.to_datetime(buys_df['timestamp'], errors='coerce', utc=True).dt.tz_localize(None)
    item_ids.update(buys_df['item_id'].dropna().astype(int).unique())
    if not buys_timestamps.dropna().empty:
        current_min, current_max = buys_timestamps.min(), buys_timestamps.max()
        if min_timestamp is None or current_min < min_timestamp: min_timestamp = current_min
        if max_timestamp is None or current_max > max_timestamp: max_timestamp = current_max
    
    del buys_df, buys_timestamps; gc.collect()

    item_encoder = LabelEncoder().fit(list(item_ids))
    category_encoder = LabelEncoder().fit(list(categories))
    
    test_split_time = max_timestamp - pd.Timedelta(days=7)
    val_split_time = test_split_time - pd.Timedelta(days=7)
    
    logger.info(f"Complete vocabulary built. Full time range: {min_timestamp} to {max_timestamp}")
    
    return item_encoder, category_encoder, val_split_time, test_split_time

def main():
    config = Config()
    os.makedirs(config.OUTPUT_PATH, exist_ok=True)
    
    item_encoder, category_encoder, val_split_time, test_split_time = fit_encoders_and_get_splits_memory_safe(config)
    
    metadata = {
        'vocab_size_items': len(item_encoder.classes_),
        'vocab_size_cats': len(category_encoder.classes_),
        'max_sequence_length': config.MAX_SEQUENCE_LENGTH,
    }
    
    logger.info(f"Initial metadata created: {metadata}")
    logger.info(f"Train period ends: {val_split_time}")
    logger.info(f"Validation period ends: {test_split_time}")

    logger.info("Streaming data to generate and write sequences to TFRecord files...")
    
    train_path = os.path.join(config.OUTPUT_PATH, 'train.tfrecord')
    val_path = os.path.join(config.OUTPUT_PATH, 'val.tfrecord')
    test_path = os.path.join(config.OUTPUT_PATH, 'test.tfrecord')

    # --- STEP 1: Initialize counters ---
    train_count, val_count, test_count = 0, 0, 0

    with tf.io.TFRecordWriter(train_path) as train_writer, \
         tf.io.TFRecordWriter(val_path) as val_writer, \
         tf.io.TFRecordWriter(test_path) as test_writer:
        
        click_iterator = pd.read_csv(os.path.join(config.DATA_PATH, config.CLICKS_FILE), header=None, names=['session_id', 'timestamp', 'item_id', 'category'], dtype={'category': str}, chunksize=config.CHUNK_SIZE)
        buys_df = pd.read_csv(os.path.join(config.DATA_PATH, config.BUYS_FILE), header=None, names=['session_id', 'timestamp', 'item_id', 'price', 'quantity'])
        buys_df['interaction_type'] = 'buy'
        
        carry_over_session = pd.DataFrame()

        for chunk in tqdm(click_iterator, desc="Processing Chunks"):
            chunk['interaction_type'] = 'click'
            chunk_sessions = chunk['session_id'].unique()
            relevant_buys = buys_df[buys_df['session_id'].isin(chunk_sessions)]
            
            merged_chunk = pd.concat([chunk, relevant_buys], ignore_index=True)
            if not carry_over_session.empty:
                merged_chunk = pd.concat([carry_over_session, merged_chunk], ignore_index=True)

            merged_chunk['timestamp'] = pd.to_datetime(merged_chunk['timestamp'], errors='coerce', utc=True).dt.tz_localize(None)
            merged_chunk.dropna(subset=['timestamp'], inplace=True)
            merged_chunk = merged_chunk.sort_values(['session_id', 'timestamp'])
            
            if merged_chunk.empty:
                continue

            last_session_id = merged_chunk['session_id'].iloc[-1]
            
            for session_id, session_df in merged_chunk.groupby('session_id'):
                if session_id == last_session_id:
                    carry_over_session = session_df
                    continue
                
                if len(session_df) < config.MIN_SEQUENCE_LENGTH: continue

                items = item_encoder.transform(session_df['item_id'].fillna(0).astype(int))
                cats = category_encoder.transform(session_df['category'].fillna('-1').astype(str))
                interactions = session_df['interaction_type'].values
                timestamps = session_df['timestamp'].values

                time_deltas = np.zeros(len(timestamps))
                time_deltas_pd = (pd.Series(timestamps[1:]) - pd.Series(timestamps[:-1]))
                time_deltas[1:] = time_deltas_pd.dt.total_seconds().astype(float)
                log_time_deltas = np.log1p(time_deltas)

                for i in range(1, len(items)):
                    label = 1 if interactions[i] == 'buy' else 0
                    start_idx = max(0, i - config.MAX_SEQUENCE_LENGTH)
                    pad_len = config.MAX_SEQUENCE_LENGTH - (i - start_idx)

                    sequence_data = {
                        'items': np.pad(items[start_idx:i], (pad_len, 0)),
                        'categories': np.pad(cats[start_idx:i], (pad_len, 0)),
                        'time_deltas': np.pad(log_time_deltas[start_idx:i], (pad_len, 0)),
                        'label': label
                    }

                    example = create_tf_example(sequence_data)
                    serialized_example = example.SerializeToString()
                    
                    event_time = timestamps[i]
                    # --- STEP 2: Increment counters on write ---
                    if event_time < val_split_time:
                        train_writer.write(serialized_example)
                        train_count += 1
                    elif event_time < test_split_time:
                        val_writer.write(serialized_example)
                        val_count += 1
                    else:
                        test_writer.write(serialized_example)
                        test_count += 1
            
    logger.info("TFRecord file generation complete.")
    logger.info(f"Train examples: {train_count:,}, Validation examples: {val_count:,}, Test examples: {test_count:,}")

    # --- STEP 3: Update and save the final metadata ---
    metadata['num_train_examples'] = train_count
    metadata['num_val_examples'] = val_count
    metadata['num_test_examples'] = test_count

    with open(os.path.join(config.OUTPUT_PATH, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    
    logger.info(f"Final metadata saved: {metadata}")

if __name__ == "__main__":
    main()