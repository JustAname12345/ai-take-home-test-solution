from typing import Union, Dict, Any, Tuple, Optional

import wandb
import torch
import torch.nn as nn
from torch import Tensor
from pytorch_lightning import LightningModule
import os
from torchvision.utils import save_image

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1024),
            nn.Tanh()  # Normalizing the output to [-1, 1]
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 1, 32, 32)  # Reshape to MNIST image dimensions
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output a single scalar per image
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

class MNISTGANModel(LightningModule):
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        #self.generator = generator
        #self.discriminator = discriminator
        #self.adversarial_loss = torch.nn.MSELoss()
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.adversarial_loss = torch.nn.BCELoss()
    def forward(self, z, labels) -> Tensor:
        return self.generator(z, labels)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.b1, self.hparams.b2),
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.b1, self.hparams.b2)
        )
        return [opt_g, opt_d], []

    def generate_and_save_images(self, batch_idx, num_images=20):
        z = torch.randn(num_images, 100, device=self.device)  # Assume the latent space dimension is 100
        with torch.no_grad():
            images = self.generator(z)
        images = (images + 1) / 2  # Rescale images from [-1, 1] to [0, 1]
        save_folder = os.path.join(os.getcwd(),'generated_images')
        os.makedirs(save_folder, exist_ok=True)
        save_path = f'{save_folder}/iter_{batch_idx}'
        os.makedirs(save_path, exist_ok=True)
        save_image(images, os.path.join(save_path, 'mnist_fake.png'), nrow=5)  # Save images in a grid of 5x4

    def training_step(self, batch, batch_idx, optimizer_idx) -> Union[Tensor, Dict[str, Any]]:
        log_dict, loss = self.step(batch, batch_idx, optimizer_idx)
        self.log_dict({"/".join(("train", k)): v for k, v in log_dict.items()})
        if batch_idx % 3 == 0:
            self.generate_and_save_images(batch_idx)
        return loss

    def validation_step(self, batch, batch_idx) -> Union[Tensor, Dict[str, Any], None]:
        log_dict, loss = self.step(batch, batch_idx)
        val_loss = log_dict['d_loss']+log_dict['g_loss']
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return {'val_loss': val_loss}
        '''
        self.log_dict({"/".join(("val", k)): v for k, v in log_dict.items()})
        return None
        '''

    def test_step(self, batch, batch_idx) -> Union[Tensor, Dict[str, Any], None]:
        # TODO: if you have time, try implementing a test step
        imgs, _ = batch  # Assuming labels aren't needed for testing
        batch_size = imgs.shape[0]
        valid = torch.ones(batch_size, 1, device=self.device)
        fake = torch.zeros(batch_size, 1, device=self.device)

        # Generate random noise
        z = torch.randn(batch_size, 100, device=self.device)
        # Generate images
        gen_imgs = self.generator(z)

        # Discriminator's ability to classify real images as real
        real_loss = self.adversarial_loss(self.discriminator(imgs), valid)
        # Discriminator's ability to classify fake images as fake
        fake_loss = self.adversarial_loss(self.discriminator(gen_imgs.detach()), fake)
        # Overall loss for discriminator
        d_loss = (real_loss + fake_loss) / 2
        # Loss for generator is not typically evaluated in test_step

        # Log test losses to some logger or simply return them
        log_dict = {'test_real_loss': real_loss, 'test_fake_loss': fake_loss, 'test_d_loss': d_loss}
        self.log_dict(log_dict)
        return log_dict

    def step(self, batch, batch_idx, optimizer_idx=None):
        imgs, _ = batch  # Assume labels aren't needed
        batch_size = imgs.shape[0]
        valid = torch.ones(batch_size, 1, device=self.device)
        fake = torch.zeros(batch_size, 1, device=self.device)

        z = torch.randn(batch_size, 100, device=self.device)  # Generate random noise
        gen_imgs = self.generator(z)

        real_loss = self.adversarial_loss(self.discriminator(imgs), valid)
        fake_loss = self.adversarial_loss(self.discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        g_loss = self.adversarial_loss(self.discriminator(gen_imgs), valid)

        loss = d_loss if optimizer_idx == 1 else g_loss
        log_dict = {'g_loss': g_loss, 'd_loss': d_loss}
        return log_dict, loss
    '''
    def step(self, batch, batch_idx, optimizer_idx=None) -> Tuple[Dict[str, Tensor], Optional[Tensor]]:
        # TODO: implement the step method of the GAN model.
        #     : This function should return both a dictionary of losses
        #     : and current loss of the network being optimised.
        #     :
        #     : When training with pytorch lightning, because we defined 2 optimizers in
        #     : the `configure_optimizers` function above, we use the `optimizer_idx` parameter
        #     : to keep a track of which network is being optimised.

        imgs, labels = batch
        batch_size = imgs.shape[0]

        log_dict = {}
        loss = None

        # TODO: Create adversarial ground truths

        # TODO: Create noise and labels for generator input

        if optimizer_idx == 0 or not self.training:
            # TODO: generate images and calculate the adversarial loss for the generator
            # HINT: when optimizer_idx == 0 the model is optimizing the generator
            raise NotImplementedError

            # TODO: Generate a batch of images

            # TODO: Calculate loss to measure generator's ability to fool the discriminator

        if optimizer_idx == 1 or not self.training:
            # TODO: generate images and calculate the adversarial loss for the discriminator
            # HINT: when optimizer_idx == 1 the model is optimizing the discriminator
            raise NotImplementedError

            # TODO: Generate a batch of images

            # TODO: Calculate loss for real images

            # TODO: Calculate loss for fake images

            # TODO: Calculate total discriminator loss

        return log_dict, loss
    '''
    def on_epoch_end(self):
        # TODO: implement functionality to log predicted images to wandb
        #     : at the end of each epoch

        # TODO: Create fake images

        for logger in self.trainer.logger:
            if type(logger).__name__ == "WandbLogger":
                # TODO: log fake images to wandb (https://docs.wandb.ai/guides/track/log/media)
                #     : replace `None` with your wandb Image object
                logger.experiment.log({"gen_imgs": None})
