# MaiMemo CEFR 前処理

`preprocess_maimemo_cefr.py` は MaiMemo のTSVを、CEFRレベル付きの Hive 形式Parquetに変換します。

## 入力データ
- `opensource_dataset.tsv` を Harvard Dataverse からダウンロード: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/VAGUL0
- ダウンロードしたTSVをこのフォルダに置く（またはスクリプト内の `MAIMEMO_TSV` を実際のパスに変更）。
- CEFR参照DB `word_cefr_minified.db` は既にこのフォルダにあります。

## 使い方
1) 必要なPythonパッケージ: `pandas`, `pyarrow`, `tqdm`
2) 入出力パスはスクリプト内で以下デフォルトになっています。必要なら編集してください。
   - 入力TSV: `/home/iv/srs_research/maimemo_datasets/opensource_dataset.tsv`
   - CEFR DB: `/home/iv/srs_research/Words-CEFR-Dataset/word_cefr_minified.db`
   - 出力先: `/home/iv/srs_research/maimemo_parquet_cefr/revlogs/user_id=XXX/data.parquet`
3) 実行例（このフォルダで実行する場合）:
   ```bash
   python preprocess_maimemo_cefr.py
   ```
   TSVをストリーム処理し、`word` と `cefr_level` を付与し、履歴を展開して Hive パーティションに書き出します。
