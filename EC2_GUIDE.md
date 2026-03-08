# EC2 GPU Instance — Usage Guide

## Instance Details

| Field | Value |
|-------|-------|
| Instance ID | `i-074325e5ca28fd72b` |
| Type | `g4dn.4xlarge` (16 vCPUs, 64 GB RAM, 1x NVIDIA T4 16GB) |
| Region | `us-west-2` |
| Private IP | `10.208.47.64` |
| OS | Ubuntu 24.04 |
| Storage | 500 GB gp3 (10K IOPS, 1000 MB/s) |
| SSH User | `ubuntu` |
| SSH Key | `terraform/gpu-g4dn/gpu-key` |
| AWS Profile | `826251693180-/DEVROLE` |

> **Note**: The private IP may change after a stop/start cycle. See [Get Updated IP](#get-updated-ip-after-restart) below.

---

## 1. SSH Setup (one-time)

Add this to `~/.ssh/config` on your Mac:

```
Host gpu
  Hostname 10.208.47.64
  User ubuntu
  IdentityFile ~/Development/Auto Core Create/finetuning/terraform/gpu-g4dn/gpu-key
  StrictHostKeyChecking no
  ServerAliveInterval 60
  ServerAliveCountMax 3
  ForwardAgent yes
```

Now you can connect with just:

```bash
ssh gpu
```

The `ServerAliveInterval` and `ServerAliveCountMax` settings keep the connection alive and auto-detect disconnects (no more frozen terminals).

---

## 2. Connecting

### Basic SSH

```bash
ssh gpu
```

### With port forwarding (Jupyter, TensorBoard, etc.)

```bash
# Jupyter on port 8888, TensorBoard on 6006
ssh -L 8888:localhost:8888 -L 6006:localhost:6006 gpu
```

Then open `http://localhost:8888` in your browser.

### Copy files to/from the instance

```bash
# Local → EC2
scp -r ./my-data gpu:~/data/

# EC2 → Local
scp gpu:~/output/model.pt ./local-output/

# For large transfers, use rsync (resumable)
rsync -avz --progress ./my-data/ gpu:~/data/
rsync -avz --progress gpu:~/output/ ./local-output/
```

---

## 3. tmux — Persistent Sessions

tmux lets your training survive SSH disconnects. This is essential for long-running jobs.

### tmux on the EC2 instance

```bash
# SSH in
ssh gpu

# Start a new named session
tmux new -s train

# Now run your training — it persists even if SSH drops
cd ~/project && python train.py

# Detach from session (keeps it running): Ctrl+B, then D

# List sessions
tmux ls

# Reattach to a session
tmux attach -t train

# Kill a session when done
tmux kill-session -t train
```

### tmux cheat sheet

| Action | Keys |
|--------|------|
| Detach (leave running) | `Ctrl+B`, then `D` |
| New window | `Ctrl+B`, then `C` |
| Next window | `Ctrl+B`, then `N` |
| Previous window | `Ctrl+B`, then `P` |
| Split horizontal | `Ctrl+B`, then `"` |
| Split vertical | `Ctrl+B`, then `%` |
| Switch pane | `Ctrl+B`, then arrow key |
| Scroll mode | `Ctrl+B`, then `[` (q to exit) |
| Kill current pane | `Ctrl+B`, then `X` |

### Typical workflow

```bash
# 1. SSH in and start tmux
ssh gpu
tmux new -s experiment

# 2. Split into panes: training + monitoring
#    Left pane: training
cd ~/project && python train.py

#    Ctrl+B, % to split vertically
#    Right pane: GPU monitoring
watch -n 1 nvidia-smi

# 3. Detach: Ctrl+B, D
# 4. Close your laptop, go to sleep, whatever
# 5. Come back later:
ssh gpu
tmux attach -t experiment
# Everything is still running exactly where you left it
```

### tmux on your local Mac (optional)

You can also wrap the SSH connection itself in a local tmux session, so even if Terminal.app crashes, you can reattach:

```bash
# On your Mac
tmux new -s remote
ssh gpu
tmux attach -t train   # attach to the remote session

# If your terminal crashes:
# Open new terminal
tmux attach -t remote   # reattach local session
# You're back in SSH, and the remote tmux is still there
```

---

## 4. Installed Software

Pre-installed via cloud-init:

- **GPU**: NVIDIA Driver 550, CUDA 12.4
- **Python**: 3.13.1 (via pyenv)
- **ML**: PyTorch (CUDA 12.4), transformers, datasets, accelerate, peft, bitsandbytes
- **Data**: numpy, pandas, scikit-learn
- **Tools**: jupyter, ipython, nvtop, htop, git
- **AWS**: boto3 with optimized S3 config (50 concurrent requests, 64MB chunks)

### Verify GPU

```bash
ssh gpu 'nvidia-smi'
# Should show Tesla T4, 15360 MiB

ssh gpu 'python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"'
# True Tesla T4
```

---

## 5. Running Embedding Experiments

### First-time setup on EC2

```bash
ssh gpu
tmux new -s autoresearch

# Clone the repo
git clone https://github.com/stabgan/autoresearch.git ~/autoresearch
cd ~/autoresearch

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Install dependencies
uv sync

# Upload your data (from your Mac, in another terminal)
# scp -r "/path/to/data/train.csv" gpu:~/autoresearch/embedding/data/
# scp -r "/path/to/data/test.csv" gpu:~/autoresearch/embedding/data/

# Prepare data
uv run embedding/prepare_embedding.py \
  --train-csv ~/autoresearch/embedding/data/train.csv \
  --test-csv ~/autoresearch/embedding/data/test.csv

# Run an experiment
cd embedding && uv run train_embedding.py
```

### Running the autonomous loop

```bash
ssh gpu
tmux attach -t autoresearch   # or tmux new -s autoresearch

cd ~/autoresearch
# Point your AI agent at embedding/program_embedding.md and let it go
# Detach with Ctrl+B, D — it keeps running
```

---

## 6. Instance Management

### Start / Stop

```bash
# Stop (saves money, keeps disk)
AWS_PROFILE="826251693180-/DEVROLE" aws ec2 stop-instances \
  --region us-west-2 --instance-ids i-074325e5ca28fd72b

# Start
AWS_PROFILE="826251693180-/DEVROLE" aws ec2 start-instances \
  --region us-west-2 --instance-ids i-074325e5ca28fd72b
```

### Check status

```bash
AWS_PROFILE="826251693180-/DEVROLE" aws ec2 describe-instance-status \
  --region us-west-2 --instance-ids i-074325e5ca28fd72b \
  --query 'InstanceStatuses[0].InstanceState.Name' --output text
```

### Get updated IP after restart

The private IP may change after stop/start. Update your SSH config:

```bash
NEW_IP=$(AWS_PROFILE="826251693180-/DEVROLE" aws ec2 describe-instances \
  --region us-west-2 --instance-ids i-074325e5ca28fd72b \
  --query 'Reservations[0].Instances[0].PrivateIpAddress' --output text)

echo "New IP: $NEW_IP"

# Update ~/.ssh/config with the new IP
sed -i '' "s/Hostname .*/Hostname $NEW_IP/" ~/.ssh/config
# (only updates the first match — if you have multiple hosts, edit manually)
```

### Destroy (permanent)

```bash
cd ~/Development/Auto\ Core\ Create/finetuning/terraform/gpu-g4dn
terraform destroy -auto-approve
```

---

## 7. Cost

| Component | Cost |
|-----------|------|
| g4dn.4xlarge compute | ~$1.20/hour |
| 500 GB gp3 storage | ~$40/month |

**Stop the instance when not in use.** Storage charges continue even when stopped, but compute stops billing.

---

## 8. Troubleshooting

### SSH connection refused
- Instance might be stopped. Start it (see above).
- IP might have changed after restart. Get the new IP (see above).
- Make sure you're on the VPN (private IP requires VPC access).

### SSH connection freezes / hangs
- The `ServerAliveInterval 60` in SSH config should prevent this.
- If it still happens, kill the frozen terminal and reconnect. Your tmux session on the EC2 is still alive.

### GPU not detected
```bash
# Check driver
nvidia-smi

# If it fails, the driver may need reinstall after a kernel update
sudo apt-get install -y nvidia-driver-550
sudo reboot
```

### Out of disk space
```bash
# Check usage
df -h

# Clean up common culprits
pip cache purge
rm -rf ~/.cache/huggingface/hub/*  # re-downloads models as needed
sudo apt-get autoremove -y
```

### tmux session lost after reboot
tmux sessions don't survive reboots. Always save your work before stopping the instance. After reboot, just create a new tmux session.
