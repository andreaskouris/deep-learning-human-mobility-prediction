"""Controlled traffic preprocessing for the six canonical experiments."""
import torch

def _validate_dynamic_matrix(dynamic_features: torch.Tensor) -> None:
    if dynamic_features.ndim != 2:
        raise ValueError('Dynamic features must be a two-dimensional tensor.')
    if not torch.is_floating_point(dynamic_features):
        raise TypeError('Dynamic features must use a floating-point dtype.')

def preprocess_controlled(dynamic_features: torch.Tensor, observed_roads: torch.Tensor) -> torch.Tensor:
    """Preserve observed zeros and impute only absent roads.

    Every entirely absent road receives the mean traffic profile of
    all observed roads, calculated independently for each time slot.
    """
    _validate_dynamic_matrix(dynamic_features)
    result = dynamic_features.clone()
    if observed_roads.ndim != 1:
        raise ValueError('Observed-road mask must be one-dimensional.')
    if observed_roads.shape[0] != result.shape[0]:
        raise ValueError('Observed-road mask length does not match road count.')
    observed_roads = observed_roads.to(device=result.device, dtype=torch.bool)
    if not bool(observed_roads.any()):
        raise ValueError('At least one road must contain observations.')
    missing_roads = ~observed_roads
    time_slot_mean = result[observed_roads].mean(dim=0)
    if bool(missing_roads.any()):
        result[missing_roads] = time_slot_mean
    if not bool(torch.isfinite(result).all()):
        raise ValueError('Dynamic features contain non-finite values.')
    min_value = torch.min(result)
    max_value = torch.max(result)
    if bool(max_value <= min_value):
        raise ValueError('Dynamic feature range must be greater than zero.')
    return (result - min_value) / (max_value - min_value)
