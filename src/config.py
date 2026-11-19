from pathlib import Path
import torch


class Config:
    # Reproducibility & Device 
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    ROOT_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = ROOT_DIR / "data"
    CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
    FIGURES_DIR = ROOT_DIR / "results" / "figures"
    METRICS_DIR = ROOT_DIR / "results" / "metrics"

    # Dataset & DataLoader (Standardized 80/10/10 Split)
    DATASET_NAME = "FashionMNIST"
    IMG_CHANNELS = 1
    IMG_SIZE = 28
    INPUT_DIM = IMG_CHANNELS * IMG_SIZE * IMG_SIZE  
    NUM_CLASSES = 10
    CLASS_NAMES = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]

    TOTAL_SAMPLES = 70000
    TRAIN_RATIO = 0.80  
    VAL_RATIO = 0.10    
    TEST_RATIO = 0.10   

    BATCH_SIZE = 128
    EVAL_BATCH_SIZE = 64 
    NUM_WORKERS = 2

    # Model 0: MLP Classifier & Frozen Decoder
    MLP_HIDDEN_DIMS = [256, 64]
    MLP_LATENT_DIM = 64
    MLP_EPOCHS = 20
    MLP_LR = 1e-3
    FROZEN_DECODER_EPOCHS = 25
    FROZEN_DECODER_LR = 1e-3

    # Model 1: Vanilla Autoencoder 
    AE_LATENT_DIM = 64
    AE_EPOCHS = 30
    AE_LR = 1e-3

    # Model 2: Sparse Autoencoder 
    SPARSE_AE_HIDDEN_DIM = 256
    SPARSE_AE_LATENT_DIM = 64
    SPARSE_L1_LAMBDA = 1e-4
    SPARSE_AE_EPOCHS = 30
    SPARSE_AE_LR = 1e-3

    # Model 3: Variational Autoencoder (VAE)
    VAE_LATENT_DIM = 64
    VAE_KL_WEIGHT = 1.0
    VAE_EPOCHS = 30
    VAE_LR = 1e-3

    # Model 4: Vanilla GAN (MLP) 
    VANILLA_GAN_NOISE_DIM = 100
    VANILLA_GAN_EPOCHS = 50
    VANILLA_GAN_LR = 2e-4
    GAN_BETAS = (0.5, 0.999)

    # Model 5: Deep Convolutional GAN (DCGAN) 
    DCGAN_NOISE_DIM = 100
    DCGAN_FEATURE_MAPS = 64
    DCGAN_EPOCHS = 50
    DCGAN_LR = 2e-4

    # Model 6: Denoising Diffusion Probabilistic Model (DDPM) 
    DDPM_TIMESTEPS = 1000
    DDPM_BETA_START = 1e-4
    DDPM_BETA_END = 0.02
    DDPM_BASE_CHANNELS = 64
    DDPM_EPOCHS = 30
    DDPM_LR = 2e-4

    # Evaluation 
    FID_NUM_SAMPLES = 2000 

    @classmethod
    def ensure_dirs(cls):
        for directory in [cls.DATA_DIR, cls.CHECKPOINT_DIR, cls.FIGURES_DIR, cls.METRICS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)