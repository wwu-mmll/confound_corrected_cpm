from enum import IntEnum, auto


class Networks(IntEnum):
    positive = 0
    negative = 1
    both = 2

class Models(IntEnum):
    connectome = 0
    covariates = 1
    full = 2
    residuals = 3
    increment = 4

class Metrics(IntEnum):
    explained_variance_score = 0
    pearson_score = 1
    mean_squared_error = 2
    mean_absolute_error = 3
