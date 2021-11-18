from .base import Optimizer
from utils.types_ import *

class VAE_Optimizer(Optimizer):
    def __init__(self, *args) -> None:
        super(VAE_Optimizer, self).__init__(*args)

    def fit(self):
        for epoch in range(self.epochs):
            for idx, (inputs, _) in enumerate(self.data_loader):
                result = self.model(inputs)
                loss = self.model.loss_function(*result)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                if idx%100 == 0:
                    print("Epoch[{}/{}] Loss: {:.3f}".format(epoch+1, self.epochs, loss.item()/self.batch_size))
                    #recon_x, _, _ = vae(fixed_x)
                    #save_image(recon_x.view(recon_x.size(0), 1, 28, 28).data.cpu(), f'reconstructed/recon_image_{epoch}_{idx}.png')



