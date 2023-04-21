import pytorch_lightning as pl
import torch
from torch.nn.functional import threshold, normalize
from statistics import mean
from tqdm import tqdm
from segment_anything.utils.transforms import ResizeLongestSide
import numpy as np
from segment_anything import SamPredictor, sam_model_registry

class SAMModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.sam_model = sam_model_registry[self.config['MODEL']['model_type']](checkpoint=self.config['MODEL']['checkpoint'])
        self.loss_fn = torch.nn.MSELoss()
        self.transform = ResizeLongestSide(self.sam_model.image_encoder.img_size)

    def forward(self, input):
        input_image = input['image']
        input_image = self.transform.apply_image(input_image)
        input_image = self.sam_model.preprocess(input_image)
        image_embedding = self.sam_model.image_encoder(input_image)

        sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
        )

        low_res_masks, iou_predictions = self.sam_model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        masks = self.sam_model.postprocess_masks(
            low_res_masks,
            input_size=input_image.shape[-2:],
            original_size=input['original_image_size'],
        )
        masks = masks > 0

        return {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                }

    def training_step(self, batch, batch_idx):
        input, gt_binary_mask = batch
        masks_dict = self(input)
        loss = self.loss_fn(masks_dict["masks"], gt_binary_mask)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input, gt_binary_mask = batch
        binary_mask = self(input)
        loss = self.loss_fn(binary_mask, gt_binary_mask)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer



