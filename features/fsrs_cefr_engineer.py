import pandas as pd
import torch
from utils import cum_concat
from .fsrs_engineer import FSRSFeatureEngineer


class FSRSCEFRFeatureEngineer(FSRSFeatureEngineer):
    """Feature engineer for FSRS-6-CEFR that adds cefr_level to the tensor."""

    def _model_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Ensure CEFR column exists and is numeric
        if "cefr_level" not in df.columns:
            df["cefr_level"] = 0
        df["cefr_level"] = df["cefr_level"].fillna(0).astype(int)

        # Build histories
        t_history_list, r_history_list = self.get_history_lists(df)
        cefr_history_list = df.groupby("card_id", group_keys=False)["cefr_level"].apply(
            lambda x: cum_concat([[i] for i in x])
        )

        # Create tensor with shape (seq_len, 3): [delta_t, rating, cefr_level]
        df["tensor"] = [
            torch.tensor((t_item[:-1], r_item[:-1], c_item[:-1]), dtype=torch.float32)
            .transpose(0, 1)
            for t_sublist, r_sublist, c_sublist in zip(
                t_history_list, r_history_list, cefr_history_list
            )
            for t_item, r_item, c_item in zip(t_sublist, r_sublist, c_sublist)
        ]

        return df
