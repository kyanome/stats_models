
class Optimizer():
    def __init__(self, *args) -> None:
        super().__init__()
        self.model = args[0]
        self.optimizer = args[1]
        self.data_loader = args[2]
        self.batch_size = args[3]
        self.epochs = args[4]
        
    def fit(self) -> None:
        raise NotImplementedError
        