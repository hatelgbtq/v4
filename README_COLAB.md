Colab quick start for v4

1) Upload or place the `v4` folder in your Google Drive (e.g. `/MyDrive/v4`).

2) Open `colab_run_v4.ipynb` in Google Colab (it's in the repo root). Follow cells:
   - Mount Drive
   - Copy `v4` into `/content/v4` (or clone from GitHub)
   - Install dependencies (`requirements.txt`)
   - Check GPU availability
   - Start training with `nohup ... > train.log 2>&1 &` (log and PID saved to Drive)

3) Tips:
   - Use `--log_dir` pointing to Drive so checkpoints persist.
   - For long runs prefer Colab Pro/Pro+ or a cloud VM to avoid session preemption.
   - If `torch.cuda.is_available()` is False but a GPU is present, install a CUDA-enabled PyTorch wheel matching the CUDA driver shown by `nvidia-smi`.

If you want I can prepare a GitHub-ready zip of `v4` or push to a repo (I will need your repo URL/credentials).