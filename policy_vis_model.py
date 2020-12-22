from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

model_filepath = Path("./policy_vis_data.pickle")


@dataclass
class VisPolicy:
    pixels: np.ndarray
    reward: float


@dataclass
class VisDataModel:
    policies: List[VisPolicy]
