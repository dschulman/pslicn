import torch.utils.data as tud
from typing import Callable, Generator, Tuple

Data = Generator[Tuple[tud.DataLoader, tud.DataLoader], None, None]
