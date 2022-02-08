# Introduction to the config file

Last updated: 08 Feb. 2022.

The configuration file must be defined by the user, and its path is the only argument to be passed for training. An example can be found under `./configs/example_config.ini`.

The file is required to test the various classes, including <mark style="background: #FFA533!important">autoencoder</mark>,
<mark style="background: #96D7FF!important">ViT</mark>,
<mark style="background: #96FF9C!important">ConvNet</mark>,
<mark style="background: #FF9696!important">ConvNeXt</mark>.

The configuration file contains 9 subsections, which are
listed below, with a general description.

| Subsection | Description |
| :---       |   :---: |
| MODEL | Model main parameters (architecture, batch size, etc) |
| AUGMENTATION | Adjustable parameters for data augmentation |
| CHECKPOINT | Rationale for model checkpoint (see `pl.callbacks.ModelCheckpoint`) |
| CRITERIA | WSI selection criteria (see `Dataloader.Dataloader.WSIQuery`)  |
| DATA | Paths to access data, relative and absolute size of datasets |
| OPTIMIZER | Selection of optimizer and relevant parameters |
| REGULARIZATION | Includes various regularization strategies |
| SCHEDULER | Selection of learning rate scheduler and adjustable parameters |
| VERBOSE | Printed details in the Python console (see `utils.GetInfo`) |

# Detailed parameters

Each of the 9 subsections contain one or more parameters whose values must be
set. Details on the parameters can be found in the comprehensive tables below.

The **Options/restrictions** column, if not empty, provides the currently available values that can
be used for the parameter. If the **Valid** column is empty, the parameter is used
for all architectures. Otherwise, the column returns the list of all architectures
that use the parameter.

## MODEL parameters

| MODEL parameters      | Description | Options/restrictions     |     Valid     |
| :---        |    :----   |          :--- | ---: |
| Activation   | Activation function for the classification head.       |  <li>"Identity", for cost functions that include the activation, such as CrossEntropyLoss.</li>       | |
| Backbone   | Basic architecture for the <mark style="background: #96FF9C!important">ConvNet</mark> class.      | Restricted to options in `torchvision.models`.      | <mark style="background: #96FF9C!important">ConvNet</mark> |
| Base_Model      | Class to use for training/inference. Will automatically set to lower case.        | <li><mark style="background: #FFA533!important">autoencoder</mark> </li><li> <mark style="background: #96D7FF!important">ViT</mark> </li><li> <mark style="background: #96FF9C!important">ConvNet</mark> </li><li> <mark style="background: #FF9696!important">ConvNeXt</mark>  | |
| Batch_Size   | Batch size for training and inference.        |       | |
| Depth   | Depth of the network (number of recursive blocks).        |       | <mark style="background: #FFA533!important">autoencoder</mark>, <mark style="background: #96D7FF!important">ViT</mark> |
| Drop_Rate   | Probability of **all** Dropout layers in architecture.  |  If 0, no Dropout layers are used.  | |
| Emb_size   | Size of the transformer patch embeddings.        | Suggested to match Sub_Patch_Size<sup>2</sup>×n_channels.     | <mark style="background: #96D7FF!important">ViT</mark> |
| Inference   | Boolean for training or inference mode.        | <li>"true" for inference mode;</li> <li> "false" for training mode. </li>      | |
| Layer_Scale   | LayerScale initial value, as implemented in [[1]](https://openaccess.thecvf.com/content/ICCV2021/html/Touvron_Going_Deeper_With_Image_Transformers_ICCV_2021_paper.html)        |       | <mark style="background: #FF9696!important">ConvNeXt</mark> |
| Loss_Function   | Model loss function.        | Restricted to options in `torch.nn`.      | |
| Max_Epochs   | Maximum number of epochs        |       | |
| Model_Save_Path   | Export directory of the model, based on the rationale used in CHECKPOINT.   |       | |
| N_Heads_ViT   | Number of heads for multihead self-attention.        |       | <mark style="background: #96D7FF!important">ViT</mark> |
| Precision   | Precision for training. Try reducing if out of memory.        | <li>16</li> <li>32</li> <li> 64</li>      | |
| Pretrained   | Boolean to use pre-trained Backbones.        | <li> "true" </li> <li> "false" </li>       | <mark style="background: #96FF9C!important">ConvNet</mark> |
| Random_Seed   | For reproducibility, implemented with `pl.seed_everything`. See source code for which modules are seeded.        |       | |
| wf   | Network parameter in the autoencoder.        |       | <mark style="background: #FFA533!important">autoencoder</mark> |

## AUGMENTATION parameters

| AUGMENTATION parameters      | Description | Options/restrictions     |     Valid         |
| :---                         |    :---:    |          ---:            |      ---:         |
| Rand_Operations              |    Integer setting the number of operations used in `transforms.RandAugment`.   |           | |

## CHECKPOINT parameters

At each epoch, the model is saved if the quantity followed in "Monitor" has reached a new "Mode" compared to previous epochs. The quantity defined in "Monitor" must be logged during training/validation/testing using the `pl.LightningModule.log()` method. For instance, to export the model when it reaches a new high in validation accuracy, set the Mode to "max" and Monitor to "val_acc_epoch".

| CHECKPOINT parameters      | Description | Options/restrictions     |     Valid     |
| :---        |    :----:   |          ---: | ---: |
| Mode        |  Value of the metric used as a decision criteria.      |           | |
| Monitor        |   Metric to monitor    |           | |

## CRITERIA parameters

For details on how the CRITERIA parameters work, see `Dataloader.Dataloader.WSIQuery`. Each CRITERIA parameter corresponds to a column in the MasterSheet (defined above as a DATA parameter). Selection of WSIs for training/validation/testing is done on the subset of WSIs meeting the criteria. If the user does not want to use a specific CRITERIA parameter, it is preferable to comment its line in the config file. Each CRITERIA parameter should be encoded as a list of one or more strings.

| CRITERIA parameters      | Description | Options/restrictions     |     Valid     |
| :---        |    :----:   |          ---: | ---: |
| Diagnosis        |  List of acceptable diagnoses.     |           | |
| id        |    List of acceptable WSI ids.   |           | |
| Type        |    List of acceptable WSI types.   |           | |

## DATA parameters

| DATA parameters      | Description | Options/restrictions     |     Valid     |
| :---        |    :----:   |          ---: | ---: |
| Dim        |    Dimension of image patches (H, W).   |     Must be a list of one or more dimensions, *e.g.* [[256, 256]]      | |
| MasterSheet        |    Path of the .csv sheet used to select WSI based on CRITERIA parameters. See `Dataloader.Dataloader.WSIQuery`.   |           | |
| N_Classes        |    Number of classes in the classification head.    |           | |
| N_Per_Sample        |    Number of tiles used per sample (WSI). If the total number of tiles per WSI is smaller than N_Per_Sample, all tiles of the WSI are used.    |           | |
| Patches_Folder        |    Path of the folder for .csv files including all tiles location and classification, for each WSI. See `TileDataset.sh` to generate such files. |           | |
| Sub_Patch_Size        |    Dimension of sub-tiles for the transformer. Each tile is divided into sub-tiles of size Sub_Patch_Size for the attention mechanism.   |           | <mark style="background: #96D7FF!important">ViT</mark> |
| SVS_Folder        |    Path of the folder containing all original WSI (.svs files)   |           | |
| Target        |    Name of the column inside the .csv files in Patches_Folder that is used as a label for training.    | <li> "sarcoma_label" </li> for sarcoma classification.           | |
| Train_Size        |    Fraction of dataset to be used for training. Splitting is done over WSIs, not individual tiles.  |           | |
| Val_Size        |    Fraction of dataset to be used for validation. Splitting is done over WSIs, not individual tiles. Fraction of test dataset is 1 - Train_Size - Val_Size.   |           | |
| Vis        |    Visibility level used. Can be a list of multiple visibility levels.   | Must be a list of one or more scalars, *e.g.* [0].          | |


## OPTIMIZER parameters

| OPTIMIZER parameters      | Description | Options/restrictions     |     Valid     |
| :---        |    :----:   |          ---: | ---: |
| Algorithm        |    Optimization algorithm.   |      Restricted to options in `torch.optim`.     | |
| eps        |    Optimisation parameter in `Adam` and `AdamW`.   |           | |
| lr        |    Learning rate.   |           | |

## REGULARIZATION parameters

| REGULARIZATION parameters      | Description | Options/restrictions     |     Valid     |
| :---        |    :----:   |          ---: | ---: |
| Label_Smoothing        |    Label smoothing value for `torch.nn.CrossEntropyLoss`.   |           | |
| Stoch_Depth        |    Stochastic depth, as proposed by [[2]](https://link.springer.com/chapter/10.1007/978-3-319-46493-0_39)   |           | <mark style="background: #FF9696!important">ConvNeXt</mark> |
| Weight_Decay        |    Weight decay, as a direct input to `torch.optim optimizers`.   |      | |

## SCHEDULER parameters

| SCHEDULER parameters      | Description | Options/restrictions     |     Valid     |
| :---        |    :----   |          :--- | ---: |
| Lin_Gamma        |    Input of `torch.optim.lr_scheduler.stepLR`   |           | If Type=='stepLR' |
| Lin_Step_Size        |    Input of `torch.optim.lr_scheduler.stepLR`   |           | If Type=='stepLR' |
| Type        |    Type of scheduler.   |  <li>'stepLR'</li> <li>'cosine_warmup'</li>         | |
| Warmup_Epochs        |    Number of warmup epochs for cosine scheduler (`transformers.optimization.get_cosine_schedule_with_warmup`)   |           | If Type=='cosine_warmup' |

## VERBOSE parameters

| VERBOSE parameters      | Description | Options/restrictions     |     Valid     |
| :---        |    :----:   |          ---: | ---: |
| Data_Info        |    Boolean to provide additional information on the dataset after data splitting. See `utils.GetInfo`.   |           | |
