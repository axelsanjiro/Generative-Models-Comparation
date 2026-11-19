from typing import Dict
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.evaluation.metrics import compute_reconstruction_metrics
from src.models.mlp_classifier import FrozenLatentDecoder, MLPClassifier

@torch.no_grad()
def run_mlp_limitation_experiments(
    classifier: MLPClassifier,
    frozen_decoder: FrozenLatentDecoder,
    test_loader: DataLoader,
    device: torch.device,
    num_classes: int = 10,
    num_vis_samples: int = 8,
) -> Dict[str, object]:

    classifier.to(device).eval()
    frozen_decoder.to(device).eval()

    all_x, all_z, all_recon = [], [], []

    for x, _ in test_loader:
        x = x.to(device)
        z_mlp = classifier.extract_features(x)
        x_frozen_recon = frozen_decoder(z_mlp)

        all_x.append(x.cpu())
        all_z.append(z_mlp.cpu())
        all_recon.append(x_frozen_recon.cpu())

    test_x = torch.cat(all_x, dim=0)
    test_z = torch.cat(all_z, dim=0)
    test_frozen_recon = torch.cat(all_recon, dim=0)

    # Stage 4 Quantitative Metrics across entire Test Set
    frozen_metrics = compute_reconstruction_metrics(test_frozen_recon, test_x)

    # Sample subset for visual inspection
    sample_x = test_x[:num_vis_samples].to(device)
    sample_z = test_z[:num_vis_samples].to(device)

    # Stage 2: Class one-hot weight transposition (all 10 classes)
    eye_classes = torch.eye(num_classes, device=device)
    class_transpose_imgs = classifier.transpose_from_class(eye_classes).cpu()

    # Stage 3: Latent weight transposition from real test z_MLP
    latent_transpose_imgs = classifier.transpose_from_latent(sample_z).cpu()

    # Stage 4: Frozen decoder reconstructions on same samples
    frozen_recon_imgs = test_frozen_recon[:num_vis_samples]

    # Stage 5: Gaussian prior sampling N(mu_z, diag(sigma_z^2)) in MLP latent space
    mu_z = test_z.mean(dim=0, keepdim=True).to(device)
    sigma_z = test_z.std(dim=0, keepdim=True).to(device)
    z_rand = mu_z + sigma_z * torch.randn((num_vis_samples, test_z.size(1)), device=device)
    prior_sampled_imgs = frozen_decoder(z_rand).cpu()

    return {
        "frozen_decoder_metrics": frozen_metrics,
        "latent_stats": {
            "mu_mean": round(mu_z.mean().item(), 4),
            "sigma_mean": round(sigma_z.mean().item(), 4),
            "sparsity_zero_ratio": round((test_z == 0).float().mean().item(), 4),
        },
        "visuals": {
            "ground_truth": sample_x.cpu(),
            "class_transpose": class_transpose_imgs,
            "latent_transpose": latent_transpose_imgs,
            "frozen_decoder_recon": frozen_recon_imgs,
            "mlp_prior_samples": prior_sampled_imgs,
        },
    }