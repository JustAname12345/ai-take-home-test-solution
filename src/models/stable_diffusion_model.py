import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import BertModel, BertTokenizer
from torchvision import transforms
import hydra
from omegaconf import DictConfig
from torchvision.models import resnet18
import wandb
from PIL import Image
import numpy as np

class StableDiffusionModel(pl.LightningModule):
    def __init__(self, learning_rate):
        super().__init__()
        self.save_hyperparameters()
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.image_encoder = resnet18(pretrained=True)
        self.image_encoder.fc = nn.Identity()
        self.image_feature_transform = nn.Linear(512, 22 * 768)
        self.decoder = nn.Transformer(d_model=768, nhead=8, num_encoder_layers=3, num_decoder_layers=3)
        self.final_layer = nn.Linear(768, 256 * 256 * 3)
        self.loss_fn = nn.MSELoss()

    def forward(self, input_ids, attention_mask, images):
        text_features = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        image_features = self.image_encoder(images)

        # 强制数据类型一致性
        text_features = text_features.float()


        # 调整图像特征的维度以适配文本特征
        image_features = self.image_feature_transform(image_features).view(-1, 22, 768)
        image_features = image_features.float()
        # 解码生成图像
        decoded_features = self.decoder(text_features, image_features)
        first_decoded_features = decoded_features[:, 0, :]
        image_pred = self.final_layer(first_decoded_features)

        return image_pred.view(-1, 3, 256, 256)


    def training_step(self, batch, batch_idx):
        images, texts = batch
        inputs = self.tokenizer(texts, return_tensors="pt", padding='max_length', truncation=True, max_length=22)
        input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
        predictions = self(input_ids.to(self.device), attention_mask.to(self.device), images.to(self.device))
        loss = self.loss_fn(predictions, images)
        self.log('train_loss', loss)

        if batch_idx % 20 == 0:  # 可以根据需要调整为记录图像的频率
            with torch.no_grad():
                generated_image = predictions[0].float()  # 取一张生成的图像
                generated_image = (generated_image + 1) / 2 if generated_image.min() < 0 else generated_image

                # 转换成 PIL 图像
                generated_image = generated_image.cpu().detach().permute(1, 2, 0).numpy()
                generated_image = np.clip(generated_image, 0, 1)  # 确保没有超出范围的值
                generated_image = Image.fromarray((generated_image * 255).astype('uint8'))

                # 使用 wandb 记录图像，加上 epoch 和 batch_idx 作为后缀
                caption = f"Epoch {self.current_epoch} Batch {batch_idx}"
                wandb.log({"Generated Images": [wandb.Image(generated_image, caption=caption)]})

        return loss

    def validation_step(self, batch, batch_idx):
        images, texts = batch
        inputs = self.tokenizer(texts, return_tensors="pt", padding='max_length', truncation=True, max_length=22)
        input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
        predictions = self(input_ids.to(self.device), attention_mask.to(self.device), images.to(self.device))
        val_loss = self.loss_fn(predictions, images)
        self.log('val_loss', val_loss, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        images, texts = batch
        inputs = self.tokenizer(texts, return_tensors="pt", padding='max_length', truncation=True, max_length=22)
        input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
        predictions = self(input_ids.to(self.device), attention_mask.to(self.device), images.to(self.device))
        test_loss = self.loss_fn(predictions, images)
        self.log('test_loss', test_loss, prog_bar=True)
        return test_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)



'''
class TextImageDataModule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size,drop_last=True)
'''