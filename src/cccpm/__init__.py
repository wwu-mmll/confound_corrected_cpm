from importlib.metadata import PackageNotFoundError, version

from cccpm.cpm_analysis import CPMAnalysis
from cccpm.models.linear_model import LinearCPM
from cccpm.models.nonlinear_models import DecisionTreeCPM, RandomForestCPM, GAMCPM
from cccpm.edge_selection import UnivariateEdgeSelection, PThreshold
from cccpm.constants import TaskType

try:
    __version__ = version("cccpm")
except PackageNotFoundError:  # package not installed (e.g. running from source)
    __version__ = "0.0.0+dev"

__all__ = [
    "CPMAnalysis",
    "LinearCPM",
    "DecisionTreeCPM",
    "RandomForestCPM",
    "GAMCPM",
    "UnivariateEdgeSelection",
    "PThreshold",
    "TaskType",
    "__version__",
]
