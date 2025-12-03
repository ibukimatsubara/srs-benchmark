import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
from tqdm.auto import tqdm


# CEFR DBのパス設定
CEFR_DB = Path("data_preparation_scripts/word_cefr_minified.db")

# グローバルなCEFRマッピング辞書（初回呼び出し時にロード）
_word_to_cefr_cache = None


def load_cefr_mapping():
    """
    Words-CEFR-Datasetから単語→CEFRレベルのマッピングを作成

    Returns:
        dict: {word: cefr_level (1-6)}
    """
    global _word_to_cefr_cache

    if _word_to_cefr_cache is not None:
        return _word_to_cefr_cache

    print("Loading CEFR database...")

    conn = sqlite3.connect(CEFR_DB)
    cursor = conn.cursor()

    # 全単語の平均レベルを取得
    query = '''
        SELECT w.word, AVG(wp.level) as avg_level
        FROM words w
        JOIN word_pos wp ON w.word_id = wp.word_id
        GROUP BY w.word
    '''

    cursor.execute(query)
    results = cursor.fetchall()
    conn.close()

    # レベルを丸めて1-6に変換
    word_to_cefr_dict = {}
    for word, level in results:
        rounded_level = round(level)
        word_to_cefr_dict[word.lower()] = max(1, min(6, rounded_level))

    print(f"✓ Loaded {len(word_to_cefr_dict):,} words with CEFR levels")

    _word_to_cefr_cache = word_to_cefr_dict
    return word_to_cefr_dict


def word_to_cefr(word):
    """
    単語をCEFRレベルに変換する

    Args:
        word: 単語（文字列）

    Returns:
        int: CEFRレベル (0-6)
             0=Not Found, 1=A1, 2=A2, 3=B1, 4=B2, 5=C1, 6=C2
    """
    mapping = load_cefr_mapping()
    return mapping.get(word.lower(), 0)  # 0 = Not Found


def load_maimemo_tsv(data_path):
    df = pd.read_csv(data_path, sep='\t')
    data = []
    for _, row in df.iterrows():
        entry = {
            'u': row['u'],
            'w': row['w'],
            'i': row['i'],
            'delta_t': row['delta_t'],
            'r': row['r'],
            't_history': row['t_history'].split(',') if pd.notna(row['t_history']) else [],
            'r_history': row['r_history'].split(',') if pd.notna(row['r_history']) else []
        }
        data.append(entry)
    return data
    # tsvを読み込む

def convert_rating(r):
    """
    MaiMemoのrating (0/1) をAnki形式 (1-4) に変換

    Args:
        r: 0=失敗, 1=成功

    Returns:
        1=Again, 2=Hard, 3=Good, 4=Easy

    変換ルール:
        0 (失敗) → 1 (Again)
        1 (成功) → 3 (Good)  # 成功時はデフォルトでGood
    """
    return 1 if r == 0 else 3


word2id={}
user2id={}

review_data = {}


def main(data_path):
    print("=" * 70)
    print("MaiMemo Data Processing with CEFR")
    print("=" * 70)
    print(f"Input file: {data_path}\n")

    # CEFR マッピングを最初に一度だけロード
    print("[1/4] Loading CEFR mapping...")
    cefr_mapping = load_cefr_mapping()
    print()

    print("[2/4] Loading MaiMemo TSV...")
    data = load_maimemo_tsv(data_path)
    print(f"      ✓ Total entries: {len(data):,}")
    print(f"      Sample data: {data[:2]}\n")

    print("[3/4] Processing entries...")
    for entry in tqdm(data, desc="      Progress"):
        word = entry['w']
        if word not in word2id:
            word2id[word] = len(word2id) + 1
        if entry['u'] not in user2id:
            user2id[entry['u']] = len(user2id) + 1

        word_id = word2id[word]
        user_id = user2id[entry['u']]
        cefr_level = cefr_mapping.get(word.lower(), 0)  # 直接辞書参照

        entry['t_history'].append(entry['delta_t'])
        entry['r_history'].append(entry['r'])

        review_rows = pd.DataFrame({
            'card_id': [word_id]*len(entry['r_history']),
            'elapsed_days': entry['t_history'],
            'rating': [convert_rating(r) for r in entry['r_history']],
            'cefr_level': [cefr_level]*len(entry['r_history'])
        })
        if user_id not in review_data:
            review_data[user_id] = review_rows
        else:
            review_data[user_id] = pd.concat([review_data[user_id], review_rows], ignore_index=True)

    print(f"\n      ✓ Processed {len(data):,} entries")
    print(f"      ✓ Unique words: {len(word2id):,}")
    print(f"      ✓ Unique users: {len(user2id):,}")
    print(f"      ✓ Total review records: {sum(len(df) for df in review_data.values()):,}\n")

    # 各user_idごとにreview_dataを保存する処理
    print("[4/4] Saving to Parquet files...")
    output_dir = "maimemo_parquet_cefr/revlogs"
    os.makedirs(output_dir, exist_ok=True)

    for user_id, df in tqdm(review_data.items(), desc="      Saving users"):
        df.to_parquet(f"{output_dir}/{user_id}.parquet")

    print(f"\n      ✓ Saved {len(review_data):,} user files to {output_dir}/")
    print("\n" + "=" * 70)
    print("✅ Processing completed successfully!")
    print("=" * 70)
    



if __name__ == "__main__":
    # --dataでinput tsvのpath
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to MaiMemo tsv data')
    args = parser.parse_args()
    data_path = args.data

    main(data_path)










