from utils.types_ import *
from torch import nn

class BaseVAE(nn.Module):
    
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def forward(self, input: Tensor) -> Tensor:
        pass

    def loss_function(self, input: Any) -> Tensor:
        pass


