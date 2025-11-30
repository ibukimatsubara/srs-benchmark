from typing import Dict, List, Optional, Union
import torch
from config import Config, ModelName
from models import FSRS6, FSRS6CEFR, FSRS_one_step


MODEL_REGISTRY: Dict[ModelName, type] = {
    "FSRS-6": FSRS6,
    "FSRS-6-cefr": FSRS6CEFR,
    "FSRS-6-one-step": FSRS_one_step,
}


def create_model(
    config: Config,
    model_params: Optional[Union[List[float], Dict[str, torch.Tensor], float]] = None,
):
    """
    Minimal factory for FSRS variants used in this trimmed setup.
    - FSRS-6 / FSRS-6-cefr expect a list of floats for w (optional).
    - FSRS-6-one-step expects a list of floats for w (optional).
    """
    model_cls = MODEL_REGISTRY[config.model_name]

    if config.model_name == "FSRS-6-one-step":
        # FSRS_one_step is not torch.nn.Module; it consumes plain weights.
        if model_params is not None and not isinstance(model_params, list):
            raise TypeError("FSRS-6-one-step expects model_params as List[float].")
        return model_cls(config, model_params if model_params is not None else model_cls.init_w)

    # Torch-based FSRS models
    if model_params is not None and not isinstance(model_params, list):
        raise TypeError(f"{config.model_name} expects model_params as List[float] or None.")
    instance = model_cls(config, model_params) if model_params is not None else model_cls(config)
    return instance.to(config.device)
