# Studi Komparatif Model Generatif: AE, Sparse AE, VAE, GAN, DCGAN, DDPM

Tugas Take-Home Deep Learning (D6421) Sesi 07: Generative Models.

Enam model generatif (Vanilla AE, Sparse AE, VAE, Vanilla GAN, DCGAN, DDPM) dan satu MLP Classifier
pendamping dilatih dan dievaluasi pada **Fashion-MNIST** dengan protokol yang sama. Repositori ini juga
memuat eksperimen pembuktian bahwa bobot dan fitur laten MLP Classifier tidak dapat dipakai untuk
merekonstruksi maupun men-generate citra.

## Protokol Eksperimen

| Aspek | Nilai |
|---|---|
| Dataset | Fashion-MNIST, train + test bawaan digabung (70.000 citra 28×28 grayscale, 10 kelas) |
| Split | 80/10/10 → 56.000 train / 7.000 validation / 7.000 test (`random_split`, generator seed 42) |
| Seed | `torch.manual_seed(42)`, `np.random.seed(42)` (`src/utils/seed.py`) |
| DataLoader | `batch_size=128`, `shuffle=True` (train), `shuffle=False` (val/test) |
| Rentang piksel | [0, 1] untuk AE/VAE/MLP; [−1, 1] untuk GAN dan DDPM, dikembalikan ke [0, 1] sebelum evaluasi |
| Metrik rekonstruksi | MSE, PSNR, SSIM (torchmetrics) pada test set |
| FID | torchmetrics, fitur InceptionV3 2048-dim, 7.000 citra test vs 1.024 sampel sintetis |
| Latensi | rata-rata 100 forward pass (setelah 20 warm-up) untuk batch 64 sampel |

## Struktur Repositori

```
├── src/
│   ├── config.py               # Seluruh hyperparameter dan path
│   ├── dataset.py              # DataLoader Fashion-MNIST dengan split 80/10/10
│   ├── models/                 # mlp_classifier, vanilla_ae, sparse_ae, vae, vanilla_gan, dcgan, ddpm
│   ├── trainers/               # train_mlp, train_ae (AE, Sparse AE, VAE), train_gan, train_ddpm
│   ├── evaluation/             # metrics (MSE/PSNR/SSIM/FID/latensi), mlp_limitations, visualizations
│   └── utils/                  # seed, io_helpers (checkpoint & JSON)
├── notebooks/
│   ├── 01_mlp_classifier_limitations.ipynb
│   ├── 02_vanilla_ae.ipynb
│   ├── 03_sparse_ae.ipynb
│   ├── 04_vae.ipynb
│   ├── 05_vanilla_gan.ipynb
│   ├── 06_dcgan.ipynb
│   ├── 07_ddpm.ipynb / 07_colab.ipynb             # DDPM (dijalankan di Google Colab, GPU T4)
│   └── 08_comparative_eval_and_figures.ipynb / 08_colab.ipynb   # FID, tabel komparatif, 5 figur
├── results/
│   ├── metrics/                # *_results.json per model, mlp_limitations_metrics.json, final_comparative_table.csv
│   └── figures/                
├── checkpoints/                # Bobot model (.pth), tidak di-commit (lihat .gitignore)
├── data/                       # Diunduh otomatis oleh torchvision, tidak di-commit
└── requirements.txt
```

## Cara Menjalankan

1. Buat virtual environment dan pasang dependensi:

   ```bash
   python -m venv .venv
   pip install -r requirements.txt
   ```

2. Jalankan notebook di `notebooks/` secara berurutan dari `01` sampai `08`. Setiap notebook menambahkan
   root repositori ke `sys.path`, mengimpor modul dari `src/`, menyimpan checkpoint ke `checkpoints/`, dan
   menyimpan metrik ke `results/metrics/`. Dataset diunduh otomatis ke `data/` saat pertama kali dijalankan.

   - `01`–`06` dapat dijalankan di CPU.
   - `07` (DDPM, T = 1000) dan `08` (FID untuk DDPM membutuhkan 1.024 sampel × 1.000 langkah) sebaiknya
     dijalankan di GPU. Versi `07_colab.ipynb` dan `08_colab.ipynb` disiapkan untuk Google Colab.
   - Notebook `08` membutuhkan seluruh checkpoint dari `01`–`07`.


## Konfigurasi Model

| Model | Arsitektur | Dim Laten | Optimizer / LR | Epoch |
|---|---|---|---|---|
| MLP Classifier | FC 784→256→64→10, ReLU | 64 (`z_MLP`) | Adam / 1e-3 | 20 |
| Frozen Latent Decoder | FC 64→256→784 di atas `z_MLP` beku | 64 | Adam / 1e-3 | 25 |
| Vanilla AE | Enc 784→256→64, Dec 64→256→784, loss MSE | 64 | Adam / 1e-3 | 30 |
| Sparse AE | Sama dengan Vanilla AE + L1 pada z (λ = 1e-4) | 64 | Adam / 1e-3 | 30 |
| VAE | Enc 784→256→(μ, log σ²), Dec 64→256→784, loss BCE + KL (β = 1) | 64 | Adam / 1e-3 | 30 |
| Vanilla GAN | G: 100→256→512→1024→784 (Tanh); D: 784→512→256→1 (Dropout 0.3) | 100 | Adam / 2e-4, β = (0.5, 0.999) | 50 |
| DCGAN | G: ConvT 100→256@7²→128@14²→1@28²; D: Conv 1→64@14²→128@7²→1; BatchNorm | 100 | Adam / 2e-4, β = (0.5, 0.999) | 50 |
| DDPM | U-Net kecil (base 64 ch, ResBlock + GroupNorm, time embedding sinusoidal), β linear 1e-4→0.02, T = 1000 | 28×28 | Adam / 2e-4 | 30 |

## Hasil

Semua metrik dihitung pada test set (7.000 citra). Sumber: `results/metrics/final_comparative_table.csv`.

| Model | Dim Laten | Train Time / Epoch (s) | Latency (ms / batch 64) | Test MSE | Test PSNR (dB) | Test SSIM | FID ↓ | Kestabilan |
|---|---|---|---|---|---|---|---|---|
| Vanilla AE | 64 | 8,13 | 0,72 | 0,008242 | 20,84 | 0,8360 | N/A | Stabil / Konvergen |
| Sparse AE | 64 (λ = 1e-4) | 11,82 | 0,58 | **0,008236** | **20,84** | 0,8356 | N/A | Stabil / Konvergen |
| VAE | 64 | 13,28 | 0,83 | 0,015321 | 18,15 | 0,7416 | 95,96 | Stabil / Konvergen |
| Vanilla GAN | 100 | 24,08 | 4,36 | N/A | N/A | N/A | 142,08 | Rentan osilasi / mode collapse |
| DCGAN | 100 | 195,65 | 41,17 | N/A | N/A | N/A | **29,64** | Cukup stabil (sensitif arsitektur) |
| DDPM | 28×28 | 44,73* | 16.435,84* | N/A | N/A | N/A | 34,09 | Sangat stabil |

\* DDPM dilatih dan diukur di GPU NVIDIA T4 (Colab); model lain di CPU. Angka waktu DDPM tidak dapat
dibandingkan langsung dengan model lain. FID memakai 1.024 sampel, sehingga nilainya cenderung lebih tinggi
dibanding protokol 10.000+ sampel; urutan peringkat antarmodel tetap informatif.

### Eksperimen Batasan MLP Classifier

| Tahap | Hasil |
|---|---|
| 1. Pelatihan MLP 784→256→64→10 | Akurasi test **88,89%** (syarat ≥ 88%) |
| 2. Proyeksi transpos dari one-hot `y_c` | Bercak abu-abu hampir seragam untuk semua kelas, tanpa bentuk objek |
| 3. Proyeksi transpos dari `z_MLP` citra uji | Pola biner jenuh, tidak berkorelasi dengan citra asli |
| 4. Frozen Latent Decoder vs Vanilla AE | MSE 0,017125 vs 0,008242 (≈2,08×); PSNR 17,66 vs 20,84 dB; SSIM 0,7124 vs 0,8360 |
| 5. Sampling `z ~ N(μ_z, diag(σ_z²))` | Gumpalan tanpa struktur; 56,1% elemen `z_MLP` bernilai nol (efek ReLU) |

### Figur

| Figur | Isi |
|---|---|
| `results/figures/fig1_reconstruction_grid.png` | Rekonstruksi 8 citra test: Ground Truth, Vanilla AE, Sparse AE, VAE |
| `results/figures/fig2_random_sampling_grid.png` | Grid 4×4 sampel acak VAE, Vanilla GAN, DCGAN, DDPM |
| `results/figures/fig3_mlp_limitations.png` | Bukti batasan MLP: proyeksi Wᵀ, Frozen Decoder vs Vanilla AE, prior sampling |
| `results/figures/fig4_latent_interpolation.png` | Interpolasi linier α ∈ {0; 0,2; …; 1} pada lima model |
| `results/figures/fig5_training_curves_and_ddpm_trajectory.png` | Kurva loss keenam model dan trajektori reverse diffusion t = 999 → 0 |

## Temuan Utama

- **Rekonstruksi:** Vanilla AE dan Sparse AE praktis identik (PSNR 20,84 dB). Dengan λ = 1e-4, rasio unit laten
  nol hanya naik dari 36,2% ke 37,0%, sehingga penalti sparsity belum mengubah fidelitas. VAE kehilangan
  sekitar 2,7 dB PSNR karena suku KL dan derau reparameterization, sebagai ganti ruang laten yang dapat di-sampling.
- **Kualitas sampel:** DCGAN (FID 29,64) dan DDPM (34,09) jauh lebih baik daripada VAE (95,96) dan Vanilla GAN
  (142,08). Vanilla GAN menghasilkan tekstur bintik dan didominasi kelas atasan (partial mode collapse).
- **Stabilitas:** kurva loss AE/VAE dan DDPM turun monoton dengan validasi berimpit. GAN berosilasi di awal
  pelatihan; BatchNorm dan konvolusi pada DCGAN meredam osilasi tersebut.
- **Generative Trilemma:** DDPM butuh 1.000 langkah (≈16,4 s per batch 64 di GPU), sekitar 400× lebih lambat
  dari DCGAN di CPU. VAE paling cepat tetapi paling buram.
- **Interpolasi:** Vanilla AE dan Sparse AE menghasilkan citra hantu di tengah lintasan (latent voids); VAE
  berubah paling halus; DCGAN tajam di ujung tetapi kabur di tengah.
- **MLP Classifier:** model P(y | x) dengan akurasi 88,89% tetap tidak memuat model P(x). Bobot transposnya
  tidak membentuk citra, fitur latennya menggandakan MSE rekonstruksi, dan ruang latennya tidak dapat di-sampling.

