#!/usr/bin/env python3
"""
MaiMemoデータにCEFRレベルを付与してParquet形式で保存

処理フロー:
1. opensource_dataset.tsv から読み込み (226M行)
2. 各単語にWords-CEFR-DatasetからCEFRレベルを取得
3. rating変換: 0/1 → 1-4 (Anki形式)
4. Hive形式でParquet保存: maimemo_parquet_cefr/revlogs/user_id=xxx/data.parquet

出力カラム:
- card_id: ユーザー内での単語ID (0から連番)
- elapsed_days: 前回復習からの経過日数
- rating: 1-4 (1=Again, 2=Hard, 3=Good, 4=Easy)
- word: 単語
- cefr_level: 0-6 (0=Not Found, 1=A1, ..., 6=C2)
"""
import pandas as pd
import sqlite3
from pathlib import Path
from tqdm.auto import tqdm
from collections import defaultdict

# パス設定
MAIMEMO_TSV = Path("/home/iv/srs_research/maimemo_datasets/opensource_dataset.tsv")
CEFR_DB = Path("/home/iv/srs_research/Words-CEFR-Dataset/word_cefr_minified.db")
OUTPUT_DIR = Path("/home/iv/srs_research/maimemo_parquet_cefr/revlogs")

# 処理設定
CHUNK_SIZE = 1_000_000  # 100万行ずつ処理
MAX_USERS = None  # None = 全ユーザー, 数値で制限可能

print("=" * 70)
print("MaiMemo → CEFR-enhanced Parquet Conversion")
print("=" * 70)
print(f"Input:  {MAIMEMO_TSV}")
print(f"Output: {OUTPUT_DIR}")
print(f"CEFR DB: {CEFR_DB}")
print()


def load_cefr_mapping():
    """
    Words-CEFR-Datasetから単語→CEFRレベルのマッピングを作成

    Returns:
        dict: {word: cefr_level (1-6)}
    """
    print("[1/4] Loading CEFR database...")

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
    word_to_cefr = {}
    for word, level in results:
        rounded_level = round(level)
        word_to_cefr[word.lower()] = max(1, min(6, rounded_level))

    print(f"      ✓ Loaded {len(word_to_cefr):,} words with CEFR levels")

    return word_to_cefr


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


def process_user_data(user_df, word_to_cefr):
    """
    1ユーザーのデータを処理

    Args:
        user_df: ユーザーの全復習記録
        word_to_cefr: 単語→CEFRマッピング

    Returns:
        DataFrame: 処理済みデータ
    """
    # 単語を文字列に変換（NaNや数値を処理）
    user_df['w_str'] = user_df['w'].fillna('UNKNOWN').astype(str)

    # 単語ごとにcard_idを割り当て
    unique_words = user_df['w_str'].unique()
    word_to_card_id = {word: idx for idx, word in enumerate(unique_words)}

    # データ変換
    processed = []
    for _, row in user_df.iterrows():
        word = row['w_str']
        processed.append({
            'card_id': word_to_card_id[word],
            'elapsed_days': int(row['delta_t']),
            'rating': convert_rating(int(row['r'])),
            'word': word,
            'cefr_level': word_to_cefr.get(word.lower(), 0)  # 0 = Not Found
        })

    return pd.DataFrame(processed)


def process_tsv_to_parquet(word_to_cefr):
    """
    TSVファイルをチャンク処理してParquetに変換
    """
    print("\n[2/4] Processing TSV file...")
    print(f"      Reading in chunks of {CHUNK_SIZE:,} rows")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ユーザーごとにデータを蓄積
    user_data_buffer = defaultdict(list)
    processed_users = 0
    total_rows = 0

    # TSVをチャンクで読み込み
    with tqdm(desc="Processing chunks", unit="rows", unit_scale=True) as pbar:
        for chunk in pd.read_csv(
            MAIMEMO_TSV,
            sep='\t',
            chunksize=CHUNK_SIZE,
            usecols=['u', 'w', 'delta_t', 'r'],  # 必要なカラムのみ
        ):
            total_rows += len(chunk)
            pbar.update(len(chunk))

            # ユーザーごとにグルーピング
            for user_id, user_df in chunk.groupby('u'):
                user_data_buffer[user_id].append(user_df)

            # バッファが溜まったらParquet保存
            if len(user_data_buffer) > 1000:  # 1000ユーザーごと
                save_users_to_parquet(user_data_buffer, word_to_cefr)
                processed_users += len(user_data_buffer)
                user_data_buffer.clear()

                if MAX_USERS and processed_users >= MAX_USERS:
                    print(f"\n      Reached max users limit: {MAX_USERS}")
                    break

    # 残りのユーザーを保存
    if user_data_buffer:
        save_users_to_parquet(user_data_buffer, word_to_cefr)
        processed_users += len(user_data_buffer)

    print(f"\n      ✓ Processed {total_rows:,} rows")
    print(f"      ✓ Saved {processed_users:,} users")


def save_users_to_parquet(user_data_buffer, word_to_cefr):
    """
    バッファに溜まったユーザーデータをParquetに保存
    """
    for user_id, user_dfs in user_data_buffer.items():
        # ユーザーの全データを結合
        user_df = pd.concat(user_dfs, ignore_index=True)

        # データ処理
        processed_df = process_user_data(user_df, word_to_cefr)

        # Hive形式で保存
        user_dir = OUTPUT_DIR / f"user_id={user_id}"
        user_dir.mkdir(parents=True, exist_ok=True)

        parquet_path = user_dir / "data.parquet"
        processed_df.to_parquet(parquet_path, index=False)


def verify_output():
    """
    出力データを検証
    """
    print("\n[3/4] Verifying output...")

    # ランダムなユーザーを1つ読み込んで確認
    user_dirs = list(OUTPUT_DIR.glob("user_id=*"))

    if not user_dirs:
        print("      ⚠️  No output files found!")
        return

    sample_user = user_dirs[0]
    df = pd.read_parquet(sample_user / "data.parquet")

    print(f"      Sample user: {sample_user.name}")
    print(f"      Total reviews: {len(df)}")
    print(f"      Columns: {df.columns.tolist()}")
    print(f"\n      First 5 rows:")
    print(df.head().to_string(index=False))

    # 統計情報
    print(f"\n      Statistics:")
    print(f"        Unique words: {df['word'].nunique()}")
    print(f"        CEFR level distribution:")
    cefr_dist = df['cefr_level'].value_counts().sort_index()
    cefr_names = {0: 'Not Found', 1: 'A1', 2: 'A2', 3: 'B1', 4: 'B2', 5: 'C1', 6: 'C2'}
    for level, count in cefr_dist.items():
        print(f"          {cefr_names.get(level, level)}: {count} ({count/len(df)*100:.1f}%)")

    print(f"        Rating distribution:")
    for rating, count in df['rating'].value_counts().sort_index().items():
        print(f"          {rating}: {count}")


def print_summary():
    """
    処理結果のサマリーを表示
    """
    print("\n[4/4] Summary")
    print("=" * 70)

    user_dirs = list(OUTPUT_DIR.glob("user_id=*"))
    total_users = len(user_dirs)

    # ランダムに10ユーザーをサンプリングして統計
    import random
    sample_size = min(10, total_users)
    sample_users = random.sample(user_dirs, sample_size)

    total_reviews = 0
    total_words = 0

    for user_dir in sample_users:
        df = pd.read_parquet(user_dir / "data.parquet")
        total_reviews += len(df)
        total_words += df['word'].nunique()

    avg_reviews = total_reviews / sample_size
    avg_words = total_words / sample_size

    print(f"Total users: {total_users:,}")
    print(f"Average reviews per user: {avg_reviews:.0f}")
    print(f"Average unique words per user: {avg_words:.0f}")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\n✅ Conversion completed successfully!")


def main():
    # 1. CEFRマッピング読み込み
    word_to_cefr = load_cefr_mapping()

    # 2. TSV → Parquet変換
    process_tsv_to_parquet(word_to_cefr)

    # 3. 出力検証
    verify_output()

    # 4. サマリー
    print_summary()


if __name__ == '__main__':
    main()
