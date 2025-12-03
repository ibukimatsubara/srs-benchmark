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


def load_maimemo_tsv(data_path, chunk_size=5000_000):
    """TSVをチャンクサイズごとに読み込むジェネレータ"""
    for chunk in pd.read_csv(data_path, sep='\t', chunksize=chunk_size):
        yield chunk
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


def flush_review_data(output_dir: Path):
    """BuffersをParquetへ追記し、メモリを解放する"""
    if not review_data:
        return

    for user_id, df in review_data.items():
        user_dir = output_dir / f"user_id={user_id}"
        user_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = user_dir / "data.parquet"

        if parquet_path.exists():
            existing_df = pd.read_parquet(parquet_path)
            df = pd.concat([existing_df, df], ignore_index=True)

        df.to_parquet(parquet_path)

    review_data.clear()


def main(data_path, chunk_size=500_000):
    print("=" * 70)
    print("MaiMemo Data Processing with CEFR")
    print("=" * 70)
    print(f"Input file: {data_path}\n")

    # CEFR マッピングを最初に一度だけロード
    print("[1/4] Loading CEFR mapping...")
    cefr_mapping = load_cefr_mapping()
    print()

    print(f"[2/4] Loading MaiMemo TSV in chunks of {chunk_size:,} rows...")
    total_entries = 0
    sample_data = []
    sample_printed = False

    print("[3/4] Processing entries...")
    progress_bar = tqdm(desc="      Progress", unit="row")
    output_dir = Path("maimemo_parquet_cefr/revlogs")
    output_dir.mkdir(parents=True, exist_ok=True)

    for chunk_data in load_maimemo_tsv(data_path, chunk_size):
        total_entries += len(chunk_data)
        for _, row in chunk_data.iterrows():
            user = row['u']
            word = row['w']

            if pd.isna(user) or pd.isna(word):
                progress_bar.update(1)
                continue

            user = str(user)
            word = str(word)

            item_id = row['i']
            delta_t = float(row['delta_t']) if pd.notna(row['delta_t']) else 0.0
            rating_value = int(float(row['r'])) if pd.notna(row['r']) else 0

            t_history = row['t_history'].split(',') if pd.notna(row['t_history']) else []
            t_history = [float(t) for t in t_history if t != '']
            if t_history:
                t_history[0] = -1.0

            r_history = row['r_history'].split(',') if pd.notna(row['r_history']) else []
            r_history = [int(float(r)) for r in r_history if r != '']

            if len(sample_data) < 2:
                sample_data.append({
                    'u': user,
                    'w': word,
                    'i': item_id,
                    'delta_t': delta_t,
                    'r': rating_value,
                    't_history': t_history.copy(),
                    'r_history': r_history.copy()
                })
            elif not sample_printed:
                print(f"      ✓ Sample data: {sample_data}\n")
                sample_printed = True

            if word not in word2id:
                word2id[word] = len(word2id) + 1
            if user not in user2id:
                user2id[user] = len(user2id) + 1

            word_id = word2id[word]
            user_id = user2id[user]
            cefr_level = cefr_mapping.get(word.lower(), 0)  # 直接辞書参照

            t_history.append(delta_t)
            r_history.append(rating_value)

            review_rows = pd.DataFrame({
                'card_id': [word_id]*len(r_history),
                'elapsed_days': t_history,
                'rating': [convert_rating(r) for r in r_history],
                'cefr_level': [cefr_level]*len(r_history)
            })
            if user_id not in review_data:
                review_data[user_id] = review_rows
            else:
                review_data[user_id] = pd.concat([review_data[user_id], review_rows], ignore_index=True)

            progress_bar.update(1)

        # チャンク終了時にParquetへ追記してバッファを解放
        flush_review_data(output_dir)

    progress_bar.close()

    if not sample_printed:
        print(f"      ✓ Sample data: {sample_data}\n")

    print(f"\n      ✓ Processed {total_entries:,} entries")
    print(f"      ✓ Unique words: {len(word2id):,}")
    print(f"      ✓ Unique users: {len(user2id):,}")
    print(f"      ✓ Total review records: {sum(len(df) for df in review_data.values()):,}\n")

    print("[4/4] Saving to Parquet files...")
    flush_review_data(output_dir)
    print(f"\n      ✓ Saved user folders to {output_dir}/")
    print("\n" + "=" * 70)
    print("✅ Processing completed successfully!")
    print("=" * 70)
    



if __name__ == "__main__":
    # --dataでinput tsvのpath
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to MaiMemo tsv data')
    parser.add_argument('--chunk-size', type=int, default=500_000, help='Rows per chunk when streaming TSV')
    args = parser.parse_args()
    data_path = args.data
    chunk_size = args.chunk_size

    main(data_path, chunk_size)
