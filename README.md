# Real-ESRGAN for Blind SISR task using PyTorch

This project implements a **Real-ESRGAN** (Realistic Enhanced Super-Resolution Generative Adversarial Network) model for **Blind SISR** (Single Image Super-Resolution) task. The primary goal is to upscale low-resolution (LR) images by a given factor (2x, 4x, 8x) to produce super-resolution (SR) images with high fidelity and perceptual quality.

This implementation is based on the paper [Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data](https://arxiv.org/pdf/2107.10833).

## Demonstration

The following images compare the standard bicubic interpolation with the output of the Real-ESRGAN model.

![Baboon comparison image](images/comparison_img_baboon.png)
![Butterfly comparison image](images/comparison_img_butterfly.png)
![Bird comparison image](images/comparison_img_bird.png)
![Man comparison image](images/comparison_img_man.png)
![PPT3 comparison image](images/comparison_img_ppt3.png)

## Key Features

This project is based on my [ESRGAN implementation](https://github.com/ash1ra/ESRGAN). The following key features represent the main upgrades implemented to transition to the Real-ESRGAN project and improve performance on real-world data:

- Implements a **High-Order Degradation Model** that applies a complex sequence of degradations (blur, resize, noise, JPEG) twice to synthesize realistic training data on the fly, replacing simple bicubic downsampling.
- Incorporates **sinc filters** into the data generation process to simulate and remove common ringing and overshoot artifacts found in real-world images.
- Replaces the standard VGG-style discriminator with a **U-Net Discriminator with Spectral Normalization**, which stabilizes training and provides pixel-level feedback for better local detail refinement.

## Model Architectures

### Generator

As a Generator, this project uses pretrained ESRGAN model with the same architecture.

```ascii
                                     Input (LR Image)
                                            |
                                            v
                        +-Input-Conv-Block-----------------------+
                        | Conv2D (9x9 kernel) (3 -> 64 channels) |
                        +----------------------------------------+
                                            |
                                            +---------------------------+
                                            |                           |
                                            v                           |
                  +-----+-23x-Residual-in-Residual-Blocks---------+     |
                  |     |-3x-Residual-Dense-Blocks----------------+     |
                  +-----+ Conv2D (3x3 kernel) (64 -> 32 channels) |     |
(Skip connections)|     | LeakyReLU                               |     | (Skip connection)
                  +-----+ Conv2D (3x3 kernel) (96 -> 32 channels) |     |
                  |     | LeakyReLU                               |     |
                  +-----+ Conv2D (3x3 kernel) (128 -> 32 channels)|     |
                  |     | LeakyReLU                               |     |
                  +-----+ Conv2D (3x3 kernel) (160 -> 32 channels)|     |
                  |     | LeakyReLU                               |     |
                  +-----+ Conv2D (3x3 kernel) (192 -> 64 channels)|     |
                  |     | * RESIDUAL_SCALING_VALUE + X            |     |
                  |     +-----------------------------------------+     |
                  |     | * RESIDUAL_SCALING_VALUE + X            |     |
                  +-----+-----------------------------------------+     |
                                            |                           |
                                            v                           |
                        +-Middle-Conv-Block-----------------------+     |
                        | Conv2D (3x3 kernel) (64 -> 64 channels) |     |
                        +-----------------------------------------+     |    
                                            |                           |
                                            +---------------------------+
                                            |
                                            v
                        +-2x-Sub-pixel-Conv-Blocks-----------------+
                        | Conv2D (3x3 kernel) (64 -> 256 channels) |
                        | PixelShuffle (h, w, 256 -> 2h, 2w, 64)   |
                        | PReLU                                    |
                        +------------------------------------------+
                                            |
                                            v
                        +-Final-Conv-Block-----------------------+
                        | Conv2D (9x9 kernel) (64 -> 3 channels) |
                        | Tanh                                   |
                        +----------------------------------------+
                                            |
                                            v
                                     Output (SR Image)
```

### Discriminator

As a Discriminator, this project uses a U-Net architecture with Spectral Normalization that performs pixel-wise assessment of images to provide local feedback for better texture refinement and training stability.

***Note:*** *The result of the model is logit, which is then passed to `BCEWithLogitsLoss` (with built-in `Sigmoid`) loss function, and therefore does not need a separate `Sigmoid` layer.*

## Datasets

### Training

The model is trained on the **DF2K_OST** (DIV2K + Flickr2K + OST) dataset. The `data_processing.py` script dynamically creates LR images from HR images using bicubic downsampling and applies random crops and augmentations (flips, rotations).

### Validation

The **DIV2K_valid** dataset is used for validation.

### Testing

The `test.py` script is configured to evaluate the trained model on standard benchmark datasets: **Set5**, **Set14**, **BSDS100**, and **Urban100**.

## Project Structure

```
.
├── checkpoints/             # Stores model weights (.safetensors) and training states
├── images/                  # Directory for inference inputs, outputs, and training plots
├── config.py                # Configures the application logger, hyperparameters and file paths
├── data_processing.py       # Defines the SRDataset class and image transformations
├── inference.py             # Script to run the model on a single image
├── models.py                # Generator, Discriminator and TruncatedVGG19 model architectures definition
├── test.py                  # Script for evaluating the model on benchmark datasets
├── train.py                 # Script for training the model
└── utils.py                 # Utility functions (metrics, checkpoints, plotting)
```

## Configuration

All hyperparameters, paths, and training settings can be configured in the `config.py` file.

Explanation of some settings:
- `INITIALIZE_WITH_ESRGAN_CHECKPOINT`: Set to `True` to use pre-trained ESRGAN weights (for `pretrain.py`).
- `LOAD_REAL_ESRNET_CHECKPOINT`: Set to `True` to resume training from the last Real-ESRNET checkpoint (for `pretrain.py`).
- `LOAD_BEST_REAL_ESRNET_CHECKPOINT`: Set to `True` to resume training from the best Real-ESRNET checkpoint (for `pretrain.py`).
- `INITIALIZE_WITH_REAL_ESRNET_CHECKPOINT`: Set to `True` to use pre-trained Real-ESRNET weights (for `train.py`).
- `LOAD_REAL_ESRGAN_CHECKPOINT`: Set to `True` to resume training from the last Real-ESRGAN checkpoint (for `train.py`).
- `LOAD_BEST_REAL_ESRGAN_CHECKPOINT`: Set to `True` to resume training from the best Real-ESRGAN checkpoint (for `train.py`).
- `TRAIN_DATASET_PATH`: Path to the train data. Can be a directory of images or a `.txt` file listing image paths.
- `VAL_DATASET_PATH`: Path to the validation data. Can be a directory of images or a `.txt` file listing image paths.
- `TEST_DATASETS_PATHS`: List of paths to the test data. Each path can be a directory of images or a `.txt` file listing image paths.
- `DEV_MOVE`: Set to `True` to use a 10% subset of the train data for quick testing.

***Note:*** *`INITIALIZE_WITH_REAL_ESRNET_CHECKPOINT` and `LOAD_REAL_ESRGAN_CHECKPOINT` or `LOAD_BEST_REAL_ESRGAN_CHECKPOINT` are mutually exclusive. If the first one is `True`, then the other two should be `False` and vice versa. If the first parameter is set to `True` and one of the second parameters is set to `True`, then the model weights will be overwritten by the second parameter.*

## Setting Up and Running the Project

### 1. Installation

1. Clone the repository:
```bash
git clone https://github.com/ash1ra/Real-ESRGAN.git
cd Real-ESRGAN
```

2. Create `.venv` and install dependencies:
```bash
uv sync
```

3. Activate a virtual environment:
```bash
# On Windows
.venv\Scripts\activate
# On Unix or MacOS
source .venv/bin/activate
```

### 2. Data Preparation

1.  [Download](https://data.vision.ee.ethz.ch/cvl/DIV2K/) the **DIV2K** datasets (`Train Data (HR images)` and `Validation Data (HR images)`).
2.  [Download](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) the **Flickr2K** dataset.
3.  [Download](https://drive.google.com/drive/folders/1LIb631GU3bOyQVTeuALesD8_eoApNniB) the **OST** datasets (`OutdoorSceneTest300/OST300_img.zip` and `OutdoorSceneTrain_v2`).
4.  [Download](https://figshare.com/articles/dataset/BSD100_Set5_Set14_Urban100/21586188) the standard benchmark datasets (**Set5**, **Set14**, **BSDS100**, **Urban100**).
5.  Create training dataset from **DIV2K**, **Flickr2K** and **OST** (both, test and train).
6.  Organize your data directory as expected by the scripts:
    ```
    data/
    ├── DF2K_OST/
    │   ├── 1.jpg
    │   └── ...
    ├── DIV2K_valid/
    │   ├── 1.jpg
    │   └── ...
    ├── Set5/
    │   ├── baboon.png
    │   └── ...
    ├── Set14/
    │   └── ...
    ...
    ```

    or
    
    ```
    data/
    ├── DF2K_OST.txt
    ├── DIV2K_valid.txt
    ├── Set5.txt
    ├── Set14.txt
    ...
    ```
    
8.  Update the paths (`TRAIN_DATASET_PATH`, `VAL_DATASET_PATH`, `TEST_DATASETS_PATHS`) in `config.py` to match your data structure.

### 3. Pre-training

1.  Adjust parameters in `config.py` as needed.
2.  Run the training script:
    ```bash
    python pretrain.py
    ```
3.  Training progress will be logged to the console and to a file in the `logs/` directory.
4.  Checkpoints will be saved in `checkpoints/`. A plot of the training metrics will be saved in `images/` upon completion.

### 4. Training

1.  Adjust parameters in `config.py` as needed.
2.  Run the training script:
    ```bash
    python train.py
    ```
3.  Training progress will be logged to the console and to a file in the `logs/` directory.
4.  Checkpoints will be saved in `checkpoints/`. A plot of the training metrics will be saved in `images/` upon completion.

### 5. Testing

To evaluate the model's performance on the test datasets:

1.  Ensure the `BEST_ESRGAN_CHECKPOINT_DIR_PATH` in `config.py` points to your trained model (e.g., `checkpoints/esrgan_best`).
2.  Run the test script:
    ```bash
    python test.py
    ```
3.  The script will print the average PSNR and SSIM for each dataset.

### 6. Inference

To upscale a single image:

1.  Place your image in the `images/` folder (or update the path).
2.  In `config.py`, set `INFERENCE_INPUT_IMG_PATH` to your image, `INFERENCE_OUTPUT_IMG_PATH` to desired location of output image, `INFERENCE_COMPARISON_IMG_PATH` to deisred location of comparison image (optional) and `BEST_REAL_ESRGAN_CHECKPOINT_DIR_PATH` to your trained model.
3.  Run the script:
    ```bash
    python inference.py
    ```
4.  The upscaled image (`sr_img_*.png`) and a comparison image (`comparison_img_*.png`) will be saved in the `images/` directory.

## Training Results

The training process is divided into two distinct stages, as recommended by the Real-ESRGAN paper. Both stages were trained on an NVIDIA RTX 4060 Ti (8 GB) with a batch size of 48 (with `GRADIENT_ACCUMULATION_STEPS = 12`, so the real batch is `48 / 12 = 4`).  

### Stage 1: Real-ESRNET Pre-Training

![Real-ESRNET model training metrics](images/real_esrnet_training_metrics.png)

The first stage involved training the **Real-ESRNET generator** (using L1 Loss) for **250 epochs**. This stage took nearly 37 hours. The final model was selected based on the epoch with the **highest validation PSNR**.  

### Stage 2: Real-ESRGAN Fine-Tuning

![Real-ESRGAN model training metrics](images/real_esrgan_training_metrics.png)

The pre-trained weights from Stage 1 were used to initialize the generator for **Real-ESRGAN fine-tuning**. This model was then trained for **107 epochs** using the full Real-ESRGAN loss (Perceptual, RaGAN, and L1) with learning rates `1e-4` and `1e-5` for Generator and Discriminator respectively. This stage took nearly 41 hours. The final model was selected based on the epoch with the **lowest validation loss**.

***Note:** For the inference process was taken checkpoint from the epoch 81.*

***Note 2:** It is important to consider that I was not able to train the model longer, because with different learning rate parameters the gradients still started to explode, and therefore it was decided to stop the training and move on to other projects and architectures.*

## Benchmark Evaluation (4x Upscaling)

The final model (`real_esrgan_best`) was evaluated on standard benchmark datasets. Metrics are calculated on the Y-channel after shaving 4px (the scaling factor) from the border.

**PSNR (dB) / SSIM Comparison**
| Dataset | Real-ESRGAN (this project)
| :--- | :---: |
| **Set5** | 26.91/0.8283
| **Set14** | 24.16/0.7139
| **BSDS100** | 23.42/0.6585
| **Urban100**| 21.70/0.7189

***Note**: It is crucial to remember that for perceptual models like Real-ESRGAN, traditional metrics (PSNR and SSIM) are not the primary measure of success. As highlighted in the original research, distortion (PSNR) and perceptual quality (human-perceived realism) are fundamentally at odds with each other. A model trained only for PSNR will score higher on these metrics but will produce overly smooth images. The final Real-ESRGAN model intentionally achieves lower PSNR/SSIM scores to produce sharp, realistic textures that look far more convincing to the human eye.*

## Visual Comparisons

The following images compare the standard bicubic interpolation with the output of the Real-ESRGAN model. I tried to use different images that would be visible difference in results with anime images, photos etc.

![Comparisson image 1](images/comparison_img_1.png)
![Comparisson image 2](images/comparison_img_2.png)
![Comparisson image 3](images/comparison_img_3.png)
![Comparisson image 4](images/comparison_img_4.png)
![Comparisson image 5](images/comparison_img_5.png)

## Acknowledgements

This implementation is based on the paper [Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data](https://arxiv.org/pdf/2107.10833).

```bibtex
@misc{wang2021realesrgantrainingrealworldblind,
      title={Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data}, 
      author={Xintao Wang and Liangbin Xie and Chao Dong and Ying Shan},
      year={2021},
      eprint={2107.10833},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2107.10833}, 
}
```

DIV2K dataset citation:

```bibtex
@InProceedings{Timofte_2018_CVPR_Workshops,
  author = {Timofte, Radu and Gu, Shuhang and Wu, Jiqing and Van Gool, Luc and Zhang, Lei and Yang, Ming-Hsuan and Haris, Muhammad and others},
  title = {NTIRE 2018 Challenge on Single Image Super-Resolution: Methods and Results},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month = {June},
  year = {2018}
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
