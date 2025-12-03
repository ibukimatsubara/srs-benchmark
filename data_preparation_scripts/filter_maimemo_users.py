"""
Filter MaiMemo users based on study criteria.

Filters users who have:
1. Studied 100 or more unique words

Additionally applies srs-benchmark filters to check data availability:
3. Rating filter (only rating 1 and 3)
4. Same-day review filter (elapsed_days > 0)
5. Max reviews per card (128)
6. Minimum reviews per user (6)
"""

import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import argparse


def process_user(user_dir_path):
    """
    Process a single user and check if they meet the filtering criteria.

    Args:
        user_dir_path: Path to user_id=XXXXXX directory

    Returns:
        tuple: (user_id or None for basic filter, user_id or None for benchmark filter, stats dict)
    """
    user_id = user_dir_path.name.replace('user_id=', '')
    stats = {
        'user_id': user_id,
        'total_reviews': 0,
        'unique_words': 0,
        'study_period': 0,
        'after_rating_filter': 0,
        'after_same_day_filter': 0,
        'after_max_reviews_filter': 0,
        'final_reviews': 0,
    }

    try:
        # 1. Load data.parquet
        df = pd.read_parquet(user_dir_path / "data.parquet")
        stats['total_reviews'] = len(df)

        # 2. Check unique word count
        unique_words = df['word'].nunique()
        stats['unique_words'] = unique_words
        if unique_words < 100:
            return None, None, stats

        # 3. Replace -1 (first review) with 0
        df['elapsed_days'] = df['elapsed_days'].replace(-1, 0)

        # 4. Calculate cumulative sum for each card_id (for stats only)
        df['cumulative_days'] = df.groupby('card_id')['elapsed_days'].cumsum()

        # 5. Get maximum cumulative day across all cards (for stats only)
        max_cumulative_day = df['cumulative_days'].max()

        # 6. Calculate study period (add 1 because we count from day 0) - stats only
        study_period = max_cumulative_day + 1
        stats['study_period'] = study_period

        basic_filter_pass = user_id

        # ========== Additional srs-benchmark filters ==========

        # 8. Rating filter: only rating 1 (Again) and 3 (Good)
        # This mimics script.py line 238: filters=[("rating", "in", [1, 3])]
        df_benchmark = df[df['rating'].isin([1, 3])].copy()
        stats['after_rating_filter'] = len(df_benchmark)

        if len(df_benchmark) == 0:
            return basic_filter_pass, None, stats

        # 9. Same-day review filter: elapsed_days > 0
        # This mimics script.py line 211, 227
        df_benchmark = df_benchmark[df_benchmark['elapsed_days'] > 0].copy()
        stats['after_same_day_filter'] = len(df_benchmark)

        if len(df_benchmark) == 0:
            return basic_filter_pass, None, stats

        # 10. Max reviews per card filter: max 128 reviews per card
        # This mimics script.py line 172: df.drop(df[df["i"] > max_seq_len * 2].index)
        # where max_seq_len = 64
        df_benchmark['review_num'] = df_benchmark.groupby('card_id').cumcount() + 1
        df_benchmark = df_benchmark[df_benchmark['review_num'] <= 128].copy()
        stats['after_max_reviews_filter'] = len(df_benchmark)

        if len(df_benchmark) == 0:
            return basic_filter_pass, None, stats

        # 11. Minimum review count: at least 6 reviews
        # This mimics script.py line 244
        stats['final_reviews'] = len(df_benchmark)
        if len(df_benchmark) < 6:
            return basic_filter_pass, None, stats

        return basic_filter_pass, user_id, stats

    except Exception as e:
        # Continue on error
        return None, None, stats


def main(data_dir, output_file, processes):
    """
    Main processing function.

    Args:
        data_dir: Path to maimemo_parquet_cefr directory
        output_file: Path to output file for filtered user IDs
        processes: Number of parallel processes to use
    """
    # 1. Get all user_id=* directories
    revlogs_dir = Path(data_dir) / "revlogs"
    user_dirs = sorted(revlogs_dir.glob("user_id=*"))

    print(f"Total users found: {len(user_dirs):,}")
    print(f"Using {processes} processes")
    print("=" * 80)

    # 2. Process in parallel
    with Pool(processes=processes) as pool:
        results = list(tqdm(
            pool.imap(process_user, user_dirs),
            total=len(user_dirs),
            desc="Filtering users"
        ))

    # 3. Extract user IDs that meet criteria
    basic_filtered = [basic for basic, benchmark, stats in results if basic is not None]
    benchmark_filtered = [benchmark for basic, benchmark, stats in results if benchmark is not None]
    all_stats = [stats for basic, benchmark, stats in results]

    # 4. Calculate statistics
    total_reviews = sum(s['total_reviews'] for s in all_stats)
    total_reviews_basic = sum(s['total_reviews'] for s in all_stats if s['user_id'] in basic_filtered)
    total_reviews_benchmark = sum(s['final_reviews'] for s in all_stats if s['user_id'] in benchmark_filtered)

    # 5. Print summary
    print("\n" + "=" * 80)
    print("FILTERING RESULTS")
    print("=" * 80)
    print(f"\n📊 Basic Filter (100+ words):")
    print(f"   Users passed: {len(basic_filtered):,} / {len(user_dirs):,}")
    print(f"   Retention rate: {len(basic_filtered)/len(user_dirs)*100:.2f}%")
    print(f"   Total reviews: {total_reviews_basic:,}")

    print(f"\n📊 + srs-benchmark Filter (rating 1&3, no same-day, max 128/card, min 6 reviews):")
    print(f"   Users passed: {len(benchmark_filtered):,} / {len(user_dirs):,}")
    print(f"   Retention rate: {len(benchmark_filtered)/len(user_dirs)*100:.2f}%")
    print(f"   Total reviews: {total_reviews_benchmark:,}")

    print(f"\n📉 Filter attrition:")
    if len(basic_filtered) > 0:
        print(f"   Basic → Benchmark: {len(benchmark_filtered)/len(basic_filtered)*100:.2f}% retention")
        print(f"   Users lost: {len(basic_filtered) - len(benchmark_filtered):,}")

    # 6. Save to files
    basic_output = output_file
    benchmark_output = output_file.replace('.txt', '_benchmark_ready.txt')

    with open(basic_output, 'w') as f:
        for user_id in basic_filtered:
            f.write(f"{user_id}\n")

    with open(benchmark_output, 'w') as f:
        for user_id in benchmark_filtered:
            f.write(f"{user_id}\n")

    print(f"\n💾 Output files:")
    print(f"   Basic filter: {basic_output}")
    print(f"   Benchmark-ready: {benchmark_output}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Filter MaiMemo users by study criteria and srs-benchmark compatibility'
    )
    parser.add_argument('--data', default='maimemo_parquet_cefr',
                       help='Data directory path (default: maimemo_parquet_cefr)')
    parser.add_argument('--output', default='filtered_users.txt',
                       help='Output file path (default: filtered_users.txt)')
    parser.add_argument('--processes', type=int, default=cpu_count(),
                       help=f'Number of parallel processes (default: {cpu_count()})')

    args = parser.parse_args()

    main(args.data, args.output, args.processes)
