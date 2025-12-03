#!/bin/bash

# 出力先のディレクトリを作成
mkdir -p maimemo_parquet_cefr_filtered/revlogs

# filtered_users.txtを読み込んでループ
while IFS= read -r user_id; do
  # 空行をスキップ
  if [ -z "$user_id" ]; then
    continue
  fi

  src="maimemo_parquet_cefr/revlogs/user_id=$user_id"
  dest="maimemo_parquet_cefr_filtered/revlogs/user_id=$user_id"

  # コピー実行
  if [ -d "$src" ]; then
    echo "Copying user_id=$user_id"
    cp -r "$src" "$dest"
  else
    echo "Warning: $src does not exist, skipping"
  fi
done < filtered_users.txt

echo "Copy completed!"
