#!/usr/bin/env python3
"""
FSRS-6-CEFR Benchmark Script

MaiMemoデータ（CEFR付き）でFSRS-6-CEFRモデルを学習・評価。

Usage:
    python script_fsrs_cefr.py --data ../maimemo_parquet_cefr

Options:
    --data PATH       データディレクトリ（default: ../maimemo_parquet_cefr）
    --processes N     並列プロセス数（default: 4）
    --dry             パラメータ最適化なし（初期値のみ）
    --file            詳細結果をJSONで保存
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import root_mean_squared_error, log_loss, roc_auc_score
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch
import argparse

# パス設定
sys.path.insert(0, os.path.dirname(__file__))
from models.fsrs_v6_cefr import FSRS6CEFR
from config import Config
from utils import catch_exceptions, cum_concat

# 引数パーサー
parser = argparse.ArgumentParser(description='FSRS-6-CEFR Benchmark')
parser.add_argument('--data', type=str, default='../maimemo_parquet_cefr',
                   help='Data directory path')
parser.add_argument('--processes', type=int, default=4,
                   help='Number of parallel processes')
parser.add_argument('--dry', action='store_true',
                   help='Dry run (no parameter optimization)')
parser.add_argument('--file', action='store_true',
                   help='Save detailed results to file')

args = parser.parse_args()

# 設定
DATA_PATH = Path(args.data)
DRY_RUN = args.dry
SAVE_FILE = args.file
PROCESSES = args.processes

# モデル設定
config = Config()
lr = 4e-2
gamma = 1
n_epoch = 5
n_splits = 5
batch_size = 512
max_seq_len = 64

torch.set_num_threads(1)

# 出力ディレクトリ
output_path = "FSRS-6-CEFR"
if DRY_RUN:
    output_path += "-dry-run"
os.makedirs(f"evaluation/{output_path}", exist_ok=True)

print("=" * 70)
print("FSRS-6-CEFR Benchmark")
print("=" * 70)
print(f"Data path: {DATA_PATH}")
print(f"Dry run: {DRY_RUN}")
print(f"Processes: {PROCESSES}")
print(f"Output: evaluation/{output_path}/")
print()


def create_time_series(df):
    """
    時系列データを作成

    Args:
        df: 生のレビューログ

    Returns:
        処理済みデータフレーム
    """
    df["review_th"] = range(1, df.shape[0] + 1)
    df.sort_values(by=["card_id", "review_th"], inplace=True)
    df["i"] = df.groupby("card_id").cumcount() + 1

    # 長すぎるシーケンスを除外
    df.drop(df[df["i"] > max_seq_len * 2].index, inplace=True)

    # delta_tを設定（elapsed_daysをそのまま使用）
    df["delta_t"] = df["elapsed_days"]

    # t_history, r_historyを作成
    t_history_list = df.groupby("card_id", group_keys=False)["delta_t"].apply(
        lambda x: cum_concat([[max(0, i)] for i in x])
    )
    r_history_list = df.groupby("card_id", group_keys=False)["rating"].apply(
        lambda x: cum_concat([[i] for i in x])
    )

    df["r_history"] = [
        ",".join(map(str, item[:-1])) for sublist in r_history_list for item in sublist
    ]
    df["t_history"] = [
        ",".join(map(str, item[:-1])) for sublist in t_history_list for item in sublist
    ]

    # tensorを作成（CEFR情報を含む）
    cefr_history_list = df.groupby("card_id", group_keys=False)["cefr_level"].apply(
        lambda x: cum_concat([[i] for i in x])
    )

    tensors = []
    for t_sublist, r_sublist, c_sublist in zip(t_history_list, r_history_list, cefr_history_list):
        for t_item, r_item, c_item in zip(t_sublist, r_sublist, c_sublist):
            # shape: [seq_len, 3] where 3 = [delta_t, rating, cefr_level]
            tensor = torch.tensor(
                [(t, r, c) for t, r, c in zip(t_item[:-1], r_item[:-1], c_item[:-1])]
            ).float()
            tensors.append(tensor)

    df["tensor"] = tensors

    # y（正解ラベル）を設定
    df["y"] = df["rating"].map(lambda x: {1: 0, 2: 1, 3: 1, 4: 1}[x])

    # elapsed_days > 0 のみを使用
    df = df[df["elapsed_days"] > 0].copy()

    return df.sort_values(by=["review_th"])


def train_model(train_set):
    """
    FSRS-6-CEFRモデルを訓練

    Args:
        train_set: 訓練データ

    Returns:
        最適化されたパラメータ（w）
    """
    from fsrs_optimizer import Optimizer, Trainer

    model = FSRS6CEFR(config)
    optimizer = Optimizer()

    if DRY_RUN:
        # パラメータ最適化なし
        return model.init_w
    else:
        # fsrs-optimizerのTrainerを使用
        trainer = Trainer(
            train_set,
            None,
            model.init_w,
            n_epoch=n_epoch,
            lr=lr,
            gamma=gamma,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
        )
        return trainer.train(verbose=False)


def predict(w_list, testsets, user_id):
    """
    テストセットで予測

    Args:
        w_list: パラメータリスト
        testsets: テストデータセットリスト
        user_id: ユーザーID

    Returns:
        予測確率、正解ラベル、評価データ
    """
    from fsrs_optimizer import FSRS, Collection

    p = []
    y = []
    evaluation = []

    for partition_weights, testset in zip(w_list, testsets):
        partition = int(testset["partition"].iloc[0]) if "partition" in testset.columns else 0
        w = partition_weights.get(partition, partition_weights.get(0, FSRS6CEFR.init_w))

        model = FSRS(w)
        testset["stability"], testset["difficulty"] = model.get_params(testset)

        stabilities = testset["stability"]
        delta_ts = testset["delta_t"]
        retentions = model.forgetting_curve(delta_ts, stabilities)

        p.extend(retentions.tolist())
        y.extend(testset["y"].tolist())

        testset["p"] = retentions
        evaluation.append(testset)

    return p, y, pd.concat(evaluation)


@catch_exceptions
def process(user_id):
    """
    1ユーザーのデータを処理

    Args:
        user_id: ユーザーID

    Returns:
        評価結果
    """
    # データ読み込み
    df_revlogs = pd.read_parquet(
        DATA_PATH / "revlogs" / f"user_id={user_id}",
        columns=["card_id", "rating", "elapsed_days", "word", "cefr_level"],
    )

    # 負のelapsed_daysを0に（同日復習）
    df_revlogs.loc[df_revlogs["elapsed_days"] < 0, "elapsed_days"] = 0

    # rating 1,3のみ使用（MaiMemoの2値を変換済み）
    df_revlogs = df_revlogs[df_revlogs["rating"].isin([1, 3])].copy()

    # 時系列データ作成
    dataset = create_time_series(df_revlogs)

    if dataset.shape[0] < 6:
        raise Exception(f"{user_id} does not have enough data.")

    # partition設定（MaiMemoは単一partition）
    dataset["partition"] = 0

    # TimeSeriesSplitで訓練・テスト分割
    w_list = []
    testsets = []
    tscv = TimeSeriesSplit(n_splits=n_splits)

    for train_index, test_index in tscv.split(dataset):
        train_set = dataset.iloc[train_index].copy()
        test_set = dataset.iloc[test_index].copy()

        partition_weights = {}

        try:
            # モデル訓練
            weights = train_model(train_set)
            partition_weights[0] = weights
        except Exception as e:
            if "inadequate" in str(e).lower():
                partition_weights[0] = FSRS6CEFR.init_w
            else:
                raise e

        w_list.append(partition_weights)
        testsets.append(test_set)

    # 予測と評価
    p, y, evaluation = predict(w_list, testsets, user_id)

    # メトリクス計算
    logloss = log_loss(y_true=y, y_pred=p, labels=[0, 1])

    try:
        auc = roc_auc_score(y_true=y, y_score=p)
    except Exception:
        auc = None

    # 簡易RMSE（bins）
    rmse_raw = root_mean_squared_error(y_true=y, y_pred=p)

    result = {
        "metrics": {
            "LogLoss": round(logloss, 6),
            "AUC": round(auc, 6) if auc else None,
            "RMSE": round(rmse_raw, 6),
        },
        "user": user_id,
        "size": len(y),
        "parameters": {
            0: list(map(lambda x: round(x, 6), w_list[-1][0]))
        },
    }

    # ファイル保存
    if SAVE_FILE:
        with open(f"evaluation/{output_path}/{user_id}.json", "w") as f:
            json.dump(result, f, indent=2)

    return result


def main():
    """メイン処理"""
    # ユーザーリスト取得
    user_dirs = sorted(list((DATA_PATH / "revlogs").glob("user_id=*")))
    user_ids = [d.name.split("=")[1] for d in user_dirs]

    print(f"Found {len(user_ids)} users")
    print(f"Processing with {PROCESSES} processes...")
    print()

    # 並列処理
    results = []
    with ProcessPoolExecutor(max_workers=PROCESSES) as executor:
        futures = {executor.submit(process, user_id): user_id for user_id in user_ids}

        with tqdm(total=len(user_ids), desc="Processing users") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    user_id = futures[future]
                    print(f"Error processing {user_id}: {e}")

                pbar.update(1)

    # 集計
    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)

    if results:
        logloss_values = [r["metrics"]["LogLoss"] for r in results]
        auc_values = [r["metrics"]["AUC"] for r in results if r["metrics"]["AUC"]]

        print(f"Processed users: {len(results)}")
        print(f"\nAverage LogLoss: {np.mean(logloss_values):.6f}")
        print(f"Median LogLoss:  {np.median(logloss_values):.6f}")

        if auc_values:
            print(f"\nAverage AUC: {np.mean(auc_values):.6f}")
            print(f"Median AUC:  {np.median(auc_values):.6f}")

        # 結果保存
        summary = {
            "model": "FSRS-6-CEFR",
            "total_users": len(results),
            "metrics": {
                "LogLoss": {
                    "mean": float(np.mean(logloss_values)),
                    "median": float(np.median(logloss_values)),
                },
                "AUC": {
                    "mean": float(np.mean(auc_values)) if auc_values else None,
                    "median": float(np.median(auc_values)) if auc_values else None,
                },
            },
        }

        with open(f"evaluation/{output_path}/summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n✅ Results saved to: evaluation/{output_path}/")
    else:
        print("No results to summarize.")


if __name__ == "__main__":
    main()
