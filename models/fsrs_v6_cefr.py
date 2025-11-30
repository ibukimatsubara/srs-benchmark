"""
FSRS-6-CEFR: FSRS v6 with CEFR-based Difficulty Initialization

CEFRレベル（A1-C2）に基づいてDifficultyの初期値を動的に決定する拡張版FSRS-6。

新規パラメータ:
    w[21]: A1レベルの初期Difficulty (default: 3.0)
    w[22]: A2レベルの初期Difficulty (default: 4.0)
    w[23]: B1レベルの初期Difficulty (default: 5.0)
    w[24]: B2レベルの初期Difficulty (default: 6.5)
    w[25]: C1レベルの初期Difficulty (default: 8.0)
    w[26]: C2レベルの初期Difficulty (default: 9.0)

使用方法:
    from models.fsrs_v6_cefr import FSRS6CEFR

    model = FSRS6CEFR(config)
    # CEFRレベルを含むデータで学習
"""
from typing import List, Optional
import torch
from torch import nn, Tensor
from models.fsrs_v6 import FSRS6, FSRS6ParameterClipper
from config import Config


class FSRS6CEFRParameterClipper(FSRS6ParameterClipper):
    """
    FSRS-6-CEFR用のパラメータクリッパー

    w[0-20]: FSRS-6の既存制約を継承
    w[21-26]: CEFR-based difficulty制約を追加
    """
    def __call__(self, module):
        # 既存のFSRS-6制約を適用
        super().__call__(module)

        if hasattr(module, "w"):
            w = module.w.data

            # CEFR-based difficulty制約
            # 各レベルの範囲を制限し、順序関係を保証
            w[21] = w[21].clamp(1, 5)     # A1: 易しい (1-5)
            w[22] = w[22].clamp(2, 6)     # A2: (2-6)
            w[23] = w[23].clamp(3, 7)     # B1: (3-7)
            w[24] = w[24].clamp(4, 8)     # B2: (4-8)
            w[25] = w[25].clamp(5, 9)     # C1: (5-9)
            w[26] = w[26].clamp(6, 10)    # C2: 難しい (6-10)

            # 順序制約の強制: A1 <= A2 <= B1 <= B2 <= C1 <= C2
            # 各レベルが前のレベル以上になるよう調整
            for i in range(22, 27):
                if w[i] < w[i-1]:
                    w[i] = w[i-1]

            module.w.data = w


class FSRS6CEFR(FSRS6):
    """
    FSRS-6-CEFR: CEFR-aware Difficulty Initialization

    CEFRレベルに基づいてDifficultyの初期値を設定することで、
    単語の客観的難易度を考慮した学習スケジューリングを実現。

    Attributes:
        init_w: 27個のパラメータ（FSRS-6の21個 + CEFR用6個）
    """

    # 初期パラメータ（27個）
    init_w = [
        # w[0-20]: FSRS-6の既存パラメータ（そのまま継承）
        0.212,    # w[0]: S0 for rating=1
        1.2931,   # w[1]: S0 for rating=2
        2.3065,   # w[2]: S0 for rating=3
        8.2956,   # w[3]: S0 for rating=4
        6.4133,   # w[4]: init_d base (従来のデフォルト)
        0.8334,   # w[5]: init_d rating factor
        3.0194,   # w[6]: difficulty decay
        0.001,    # w[7]: stability after failure
        1.8722,   # w[8]: stability after success factor
        0.1666,   # w[9]: next difficulty factor
        0.796,    # w[10]: mean reversion factor
        1.4835,   # w[11]: stability factor 1
        0.0614,   # w[12]: stability factor 2
        0.2629,   # w[13]: stability factor 3
        1.6483,   # w[14]: stability factor 4
        0.6014,   # w[15]: stability factor 5
        1.8729,   # w[16]: init_d mean reversion target
        0.5425,   # w[17]: short term stability factor
        0.0912,   # w[18]: short term rating adjustment
        0.0658,   # w[19]: short term stability decay
        0.1542,   # w[20]: forgetting curve decay

        # w[21-26]: CEFR-based initial difficulty（新規）
        3.0,      # w[21]: A1 - 最も易しい
        4.0,      # w[22]: A2
        5.0,      # w[23]: B1
        6.5,      # w[24]: B2
        8.0,      # w[25]: C1
        9.0,      # w[26]: C2 - 最も難しい
    ]

    # パラメータの標準偏差（正則化用）
    default_params_stddev_tensor = torch.tensor(
        [
            # w[0-20]: FSRS-6の標準偏差（継承）
            6.43, 9.66, 17.58, 27.85, 0.57, 0.28, 0.6, 0.12, 0.39, 0.18,
            0.33, 0.3, 0.09, 0.16, 0.57, 0.25, 1.03, 0.31, 0.32, 0.14, 0.27,

            # w[21-26]: CEFR用の標準偏差（推定値）
            1.0,      # w[21]: A1の標準偏差
            1.0,      # w[22]: A2
            1.0,      # w[23]: B1
            1.0,      # w[24]: B2
            1.0,      # w[25]: C1
            1.0,      # w[26]: C2
        ]
    )

    def __init__(self, config: Config, w: Optional[List[float]] = None):
        """
        初期化

        Args:
            config: 設定オブジェクト
            w: 初期パラメータ（Noneの場合はinit_wを使用）
        """
        # FSRS6 の初期化を利用しつつ 27 パラメータを渡す
        super().__init__(config, w if w is not None else self.init_w)

        # CEFR 専用のクリッパーに差し替え、初期テンソルも更新
        self.clipper = FSRS6CEFRParameterClipper(config)
        self.init_w_tensor = self.w.data.clone().to(self.config.device)
        self.gamma = 1  # L2正則化係数

    def init_d_with_cefr(self, rating: Tensor, cefr_level: Tensor) -> Tensor:
        """
        CEFR-aware difficulty initialization

        CEFRレベルに基づいてDifficultyの初期値を決定。
        CEFR情報がない場合（cefr_level=0）は従来のFSRS-6ロジックを使用。

        Args:
            rating: 復習評価 (1-4)
                1=Again, 2=Hard, 3=Good, 4=Easy
            cefr_level: CEFRレベル (0-6)
                0=Not Found, 1=A1, 2=A2, 3=B1, 4=B2, 5=C1, 6=C2

        Returns:
            初期Difficulty (1-10の範囲)

        ロジック:
            1. CEFRレベルに応じたベースDifficultyを取得 (w[21-26])
            2. ratingに応じた微調整を適用
            3. CEFR情報がない場合は従来の計算式を使用
        """
        # CEFRレベル別のベースDifficulty
        # w[21]=A1, w[22]=A2, ..., w[26]=C2
        cefr_base_difficulties = torch.stack([
            self.w[21],  # A1: 易しい
            self.w[22],  # A2
            self.w[23],  # B1
            self.w[24],  # B2
            self.w[25],  # C1
            self.w[26],  # C2: 難しい
        ])

        # CEFR情報がない場合（cefr_level=0）のフォールバック
        # 従来のFSRS-6の計算式を使用
        fallback_d = self.w[4] - torch.exp(self.w[5] * (rating - 1)) + 1

        # CEFRレベルに応じたベースDifficultyを選択
        # cefr_level: 0=not found, 1=A1, 2=A2, ..., 6=C2
        # インデックス: A1=0, A2=1, ..., C2=5
        cefr_indices = (cefr_level - 1).clamp(0, 5).long()

        # マスクを使用: cefr_level > 0 の場合のみCEFRベースを使用
        has_cefr = cefr_level > 0
        base_d = torch.where(
            has_cefr,
            cefr_base_difficulties[cefr_indices],
            fallback_d
        )

        # ratingによる微調整
        # rating=3(Good)が基準、Easy(4)で易しく、Hard(2)/Again(1)で難しく
        # w[5]を使って調整の強さを制御
        rating_adjustment = self.w[5] * (rating - 3)

        # 最終Difficulty = ベース + rating調整
        new_d = base_d + rating_adjustment

        # 1-10の範囲にクランプ
        return new_d.clamp(1, 10)

    def step(self, X: Tensor, state: Tensor) -> Tensor:
        """
        1ステップの状態更新（FSRS-6を拡張）

        初回学習時のみCEFR-based difficultyを使用。
        2回目以降は通常のFSRS-6ロジックを継承。

        Args:
            X: shape[batch_size, 2+], X[:,0]=elapsed_time, X[:,1]=rating
               X[:,2]=cefr_level (optional, 存在する場合)
            state: shape[batch_size, 2], state[:,0]=stability, state[:,1]=difficulty

        Returns:
            新しいstate: shape[batch_size, 2]
        """
        # 初回学習かどうかを判定
        if torch.equal(state, torch.zeros_like(state)):
            # 初回学習: CEFRレベルを使用
            keys = torch.tensor([1, 2, 3, 4], device=self.config.device)
            keys = keys.view(1, -1).expand(X[:, 1].long().size(0), -1)
            index = (X[:, 1].long().unsqueeze(1) == keys).nonzero(as_tuple=True)

            # Stability初期化（FSRS-6と同じ）
            new_s = torch.ones_like(state[:, 0], device=self.config.device)
            new_s[index[0]] = self.w[index[1]]

            # Difficulty初期化（CEFR-aware）
            if X.shape[1] > 2:
                # cefr_levelが提供されている場合
                cefr_level = X[:, 2]
                new_d = self.init_d_with_cefr(X[:, 1], cefr_level)
            else:
                # cefr_levelがない場合は従来のロジック
                new_d = self.init_d(X[:, 1])

            new_d = new_d.clamp(1, 10)
        else:
            # 2回目以降: 通常のFSRS-6ロジックを使用
            r = self.forgetting_curve(X[:, 0], state[:, 0], -self.w[20])
            short_term = X[:, 0] < 1
            success = X[:, 1] > 1

            new_s = torch.where(
                short_term,
                self.stability_short_term(state, X[:, 1]),
                torch.where(
                    success,
                    self.stability_after_success(state, r, X[:, 1]),
                    self.stability_after_failure(state, r),
                ),
            )
            new_d = self.next_d(state, X[:, 1])
            new_d = new_d.clamp(1, 10)

        new_s = new_s.clamp(self.config.s_min, 36500)
        return torch.stack([new_s, new_d], dim=1)
