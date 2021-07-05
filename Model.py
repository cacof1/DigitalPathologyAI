import matplotlib.pyplot as plt
from pytorch_lightning import LightningDataModule, LightningModule
import numpy as np
import torch
import CLAM
print(dir(CLAM))

"""
### Data Loader
class DataModule(LightningDataModule):
  def __init__(self, image_array, mask_array, transform):
    super().__init__()
    self.image_array = image_array
    self.mask_array = mask_array
    self.transform = transform

  def setup(self, stage: Optional[str] = None):

  def train_dataloader(self):

  def val_dataloader(self):

  def test_dataloader(self):    

## Model
class ModelRegression(LightningModule):
    def __init__(self) -> None:
      super().__init__()
      from InnerEye.ML.models.architectures.classification.image_encoder_with_mlp import ImageEncoderWithMlp
      self.model = ImageEncoderWithMlp()  
      self.test_mse: List[torch.Tensor] = []
      
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
      return self.model(x)
    
    def training_step(self, batch: Dict[str, torch.Tensor], *args: Any, **kwargs: Any) -> torch.Tensor:  # type: ignore
      input = batch["x"]
      target = batch["y"]
      prediction = self.forward(input)
      loss = 0#torch.nn.functional.mse_loss(prediction, target)
      self.log("loss", loss, on_epoch=True, on_step=False)
      return loss      

    def validation_step(self, batch: Dict[str, torch.Tensor], *args: Any, **kwargs: Any) -> torch.Tensor:  # type: ignore
      return loss      
    

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[_LRScheduler]]:
      optimizer = Adam(self.parameters(), lr=1e-1)
      scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
      return [optimizer], [scheduler]

    def on_test_epoch_start(self) -> None:
      self.test_mse = []

        
    def on_test_epoch_end(self) -> None:
        average_mse = torch.mean(torch.stack(self.test_mse))
        Path("test_mse.txt").write_text(str(average_mse.item()))


## Main

"""
