# Studi Komparatif Model Generatif: AE, Sparse AE, VAE, GAN, DCGAN, DDPM

Tugas Take-Home Deep Learning (D6421) Sesi 07: Generative Models.

Enam model generatif (Vanilla AE, Sparse AE, VAE, Vanilla GAN, DCGAN, DDPM) dan satu MLP Classifier
pendamping dilatih pada **Fashion-MNIST** dengan split 80/10/10 (56.000 / 7.000 / 7.000),
`batch_size=128`, dan seed tetap `42`.

## Isi Repositori

| Berkas | Isi |
|---|---|
| `main.ipynb` | Seluruh pipeline: data, 6 model + MLP Classifier, training, evaluasi, 5 figur wajib |
| `requirements.txt` | Dependensi Python |
| `outputs/logs/` | Riwayat loss (JSON), tabel metrik (CSV), ringkasan hasil |
| `outputs/figures/` | Figur 1-5 laporan dan figur pendukung (PNG 300 dpi) |

## Menjalankan di Google Colab (disarankan)

1. Buka `main.ipynb` di Colab, pilih `Runtime → Change runtime type → T4 GPU`.
2. `Runtime → Run all`. Sel pertama menginstal `torchmetrics[image]` dan me-mount Google Drive.
3. Hasil disimpan di `MyDrive/DL_Sesi07/outputs/`. Jika runtime terputus, jalankan ulang dari atas:
   model yang sudah dilatih dimuat dari checkpoint.

## Menjalankan Lokal

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/macOS: source .venv/bin/activate)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
jupyter notebook main.ipynb
```

## Opsi di Notebook

- `QUICK_RUN = True`: smoke test (1 epoch, 20 batch, DDPM T=50) untuk memeriksa pipeline, sekitar 5 menit.
- `RETRAIN = True`: latih ulang walaupun checkpoint sudah ada. Hapus `outputs/logs/eval_cache.json`
  setelah melatih ulang agar FID dan latency dihitung ulang.
