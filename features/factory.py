from .base import BaseFeatureEngineer
from .fsrs_engineer import FSRSFeatureEngineer
from .fsrs_one_step_engineer import FSRSOneStepFeatureEngineer
from .fsrs_cefr_engineer import FSRSCEFRFeatureEngineer
from config import Config, ModelName
from typing import Type, get_args


FEATURE_ENGINEER_REGISTRY: dict[ModelName, Type[BaseFeatureEngineer]] = {
    "FSRS-6": FSRSFeatureEngineer,
    "FSRS-6-cefr": FSRSCEFRFeatureEngineer,
    "FSRS-6-one-step": FSRSOneStepFeatureEngineer,
}


def create_feature_engineer(config: Config) -> BaseFeatureEngineer:
    """
    Factory function to create the appropriate feature engineer based on model name from config

    Args:
        config: Configuration object containing model_name and other settings

    Returns:
        Appropriate feature engineer instance

    Raises:
        ValueError: If config.model_name is not supported
    """
    model_name = config.model_name

    # Create and return the appropriate feature engineer
    feature_engineer_cls = FEATURE_ENGINEER_REGISTRY[model_name]
    return feature_engineer_cls(config)


def get_supported_models() -> tuple[str, ...]:
    """
    Get list of all supported model names

    Returns:
        List of supported model names
    """
    return get_args(ModelName)
