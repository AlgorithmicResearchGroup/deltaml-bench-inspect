# Vivaria Setup Guide with Tailscale HTTPS

Complete setup guide for deploying Vivaria with GPU support and Tailscale HTTPS access for remote access without SSH tunneling.

## System Requirements

- **OS**: Ubuntu 22.04 (tested on Lambda Labs GPU instance)
- **GPUs**: 8x NVIDIA H100 80GB (or any NVIDIA GPU setup)
- **Docker**: Installed with NVIDIA Container Toolkit
- **Tailscale**: Will be installed during setup
- **Prerequisites**:
  - Docker and Docker Compose installed
  - NVIDIA drivers and CUDA toolkit installed
  - NVIDIA Container Toolkit configured
  - API keys for OpenAI and/or Anthropic

## Overview

This guide walks through setting up Vivaria with:
- GPU support for all 8 H100 GPUs
- Tailscale HTTPS for secure remote access
- Docker Compose deployment
- RE-Bench task configuration

## Step 1: Generate Base Configuration

Navigate to the Vivaria directory and run the setup script:

```bash
cd /path/to/vivaria
./scripts/setup-docker-compose.sh
```

**What this does:**
- Creates `.env.server` with randomly generated passwords and tokens
- Creates `.env.db` with database credentials
- Sets default agent resource limits (1 CPU, 4GB RAM)

## Step 2: Configure Docker GID

Get your Docker group GID and add it to the configuration:

```bash
# Get Docker GID
getent group docker
# Output example: docker:x:999:

# Add to .env.server
echo "VIVARIA_DOCKER_GID=999" >> .env.server
```

**Why this matters:** Vivaria needs to spawn Docker containers for agents. Without the correct GID, it can't access the Docker socket.

## Step 3: Copy RE-Bench GPU Override

Copy the RE-Bench Docker Compose override file:

```bash
cp /path/to/deltamlbench/setup/docker-compose.override.yml /path/to/vivaria/
```

**What this provides:**
- GPU support configuration for all services
- `NON_INTERVENTION_FULL_INTERNET_MODELS: .+` (allows all models in full-internet tasks)
- `SKIP_SAFETY_POLICY_CHECKING: true`
- User GID configuration for Docker socket access

## Step 4: Add API Keys

Edit `.env.server` and add your API keys:

```bash
# Required: At least one LLM provider
OPENAI_API_KEY=sk-proj-...
ANTHROPIC_API_KEY=sk-ant-api03-...

# Optional but recommended for some tasks
GEMINI_API_KEY=...
```

## Step 5: Create RE-Bench Secrets File

Create `deltamlbench/secrets.env` with task-specific API keys:

```bash
# At /path/to/deltamlbench/secrets.env
AI_RD_RUST_CODECONTESTS_INFERENCE_OPENAI_API_KEY=sk-proj-...
REPLICATE_API_TOKEN=...  # Optional, for some tasks
```

## Step 6: Fix Caddyfile for HTTP (Critical)

**⚠️ GOTCHA #1: HTTPS to HTTP Mismatch**

The default Caddyfile is configured to serve HTTPS with old Tailscale certificates, but Tailscale Serve expects to proxy to HTTP.

Edit `vivaria/Caddyfile` and remove the TLS configuration:

```caddyfile
# BEFORE (wrong):
:4000 {
    tls /certs/132-145-140-184.tailf03613.ts.net.crt /certs/132-145-140-184.tailf03613.ts.net.key
    handle /api/* {
        ...
    }
}

# AFTER (correct):
:4000 {
    handle /api/* {
        ...
    }
}
```

**Why:** Tailscale Serve handles HTTPS termination externally. If Caddy serves HTTPS internally, you'll get "Client sent an HTTP request to an HTTPS server" errors.

## Step 7: Install Tailscale

Install Tailscale on the server:

```bash
curl -fsSL https://tailscale.com/install.sh | sh
```

## Step 8: Authenticate Tailscale

Connect the machine to your Tailnet:

```bash
sudo tailscale up
```

Visit the authentication URL provided and log in. Verify connection:

```bash
sudo tailscale status
hostname  # Note your hostname (e.g., 192-222-53-194)
```

## Step 9: Start Vivaria Services

**⚠️ GOTCHA #2: Start BEFORE Setting Up Tailscale Serve**

You must start the Docker containers first, otherwise Tailscale Serve will bind to ports 4000/4001 and prevent containers from starting.

```bash
cd /path/to/vivaria
sudo docker compose -f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.override.yml up --detach --wait
```

**Expected output:** All containers should start and become healthy.

**⚠️ GOTCHA #3: Containers Show as "Unhealthy" Initially**

Don't panic if containers show as "unhealthy" on first start. This is usually because:
- GPU access isn't fully initialized yet
- Health checks include `nvidia-smi` which takes time to initialize
- The containers need a restart after GPU drivers are loaded

## Step 10: Verify Container Status

Check that all services are running:

```bash
sudo docker compose ps
```

Expected containers:
- `vivaria-server-1` (API server)
- `vivaria-background-process-runner-1` (task processor)
- `vivaria-database-1` (PostgreSQL)
- `vivaria-ui-1` (web interface)

Verify GPU access inside containers:

```bash
sudo docker exec vivaria-server-1 nvidia-smi -L
sudo docker exec vivaria-background-process-runner-1 nvidia-smi -L
```

**⚠️ GOTCHA #4: GPU Access Issues**

If `nvidia-smi` fails inside containers with "Failed to initialize NVML", the containers were started before GPU configuration was fully applied. Fix:

```bash
# Stop all containers
sudo docker compose down

# Restart with full GPU configuration
sudo docker compose -f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.override.yml up --detach --wait

# Verify GPU access again
sudo docker exec vivaria-server-1 nvidia-smi -L
```

## Step 11: Configure Tailscale Serve for HTTPS

Now that containers are running, set up Tailscale HTTPS proxying:

```bash
sudo tailscale serve --bg --https 4000 localhost:4000
sudo tailscale serve --bg --https 4001 localhost:4001
```

Verify the configuration:

```bash
sudo tailscale serve status
```

Expected output:
```
https://192-222-53-194.tailf03613.ts.net:4000 (tailnet only)
|-- / proxy http://localhost:4000

https://192-222-53-194.tailf03613.ts.net:4001 (tailnet only)
|-- / proxy http://localhost:4001
```

**Note:** Replace `192-222-53-194` with your actual Tailscale hostname.

## Step 12: Get Authentication Tokens

Your access tokens are in `.env.server`:

```bash
cat .env.server | grep -E "ACCESS_TOKEN=|ID_TOKEN="
```

- **ID_TOKEN**: Use this to log into the web UI
- **ACCESS_TOKEN**: Use this for the viv CLI

## Step 13: Configure viv CLI (On Your Laptop)

On your local machine (laptop), install and configure the viv CLI:

```bash
# Install viv CLI
pip install viv-cli

# Configure with Tailscale HTTPS URLs
viv config set apiUrl https://192-222-53-194.tailf03613.ts.net:4001
viv config set uiUrl https://192-222-53-194.tailf03613.ts.net:4000

# Set the access token
viv config set evalsToken <ACCESS_TOKEN_from_env.server>
```

**⚠️ GOTCHA #5: Swapped URLs**

Make sure you have the correct ports:
- **apiUrl** = port **4001** (API server)
- **uiUrl** = port **4000** (web UI)

If you swap these, you'll get 404 errors when running tasks.

## Step 14: Test the Setup

Access the web UI from your laptop:

```
https://192-222-53-194.tailf03613.ts.net:4000
```

Log in with the **ID_TOKEN** from `.env.server`.

Run a test task from your laptop:

```bash
viv run \
  --agent-path ./modular-public-claude \
  --task-family-path ./deltamlbench/pwc_5_datasets_code_cl \
  --max-tokens 10000000 \
  --max-actions 3000 \
  pwc_5_datasets_code_cl/main
```

## Common Issues and Solutions

### Issue: Tasks Stuck in Queue

**Symptom:** Jobs are submitted but never start executing.

**Cause:** Vivaria containers don't have GPU access or Docker socket access.

**Solution:**
```bash
# Check GPU access
sudo docker exec vivaria-server-1 nvidia-smi -L
sudo docker exec vivaria-background-process-runner-1 nvidia-smi -L

# If GPU access fails, restart containers
sudo docker compose down
sudo docker compose -f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.override.yml up --detach --wait
```

### Issue: "Client sent an HTTP request to an HTTPS server"

**Symptom:** Web UI shows this error when accessing through Tailscale.

**Cause:** Caddyfile is configured to serve HTTPS, but Tailscale Serve expects HTTP.

**Solution:** Remove the `tls` line from `vivaria/Caddyfile` (see Step 6) and restart:
```bash
sudo docker compose restart ui
```

### Issue: Port Already in Use

**Symptom:** Can't start containers, error about port 4000 or 4001 already in use.

**Cause:** Tailscale Serve is already bound to those ports.

**Solution:**
```bash
# Stop Tailscale Serve
sudo tailscale serve --https=4000 off
sudo tailscale serve --https=4001 off

# Start containers
sudo docker compose up --detach --wait

# Re-enable Tailscale Serve
sudo tailscale serve --bg --https 4000 localhost:4000
sudo tailscale serve --bg --https 4001 localhost:4001
```

### Issue: 404 When Running Tasks

**Symptom:** `Request to https://...:4000/uploadFiles failed with 404`

**Cause:** You have the apiUrl and uiUrl swapped in viv config.

**Solution:**
```bash
viv config set apiUrl https://192-222-53-194.tailf03613.ts.net:4001  # API = 4001
viv config set uiUrl https://192-222-53-194.tailf03613.ts.net:4000   # UI = 4000
```

### Issue: Containers Show as Unhealthy

**Symptom:** `docker compose ps` shows containers as "unhealthy".

**Possible Causes:**
1. GPU health checks failing - restart containers (see above)
2. Services not fully started - wait 30-60 seconds
3. API endpoints not responding - check logs:
   ```bash
   sudo docker compose logs server --tail 50
   sudo docker compose logs background-process-runner --tail 50
   ```

### Issue: Can't Access Web UI After Setup

**Symptom:** Web UI not loading, connection refused.

**Solution:**
```bash
# Check Tailscale Serve is running
sudo tailscale serve status

# Test local access
curl -I http://localhost:4000

# If local access works but Tailscale doesn't, restart Tailscale Serve
sudo tailscale serve --https=4000 off
sudo tailscale serve --bg --https 4000 localhost:4000
```

## Key Configuration Files

### .env.server
Location: `vivaria/.env.server`

Key variables:
- `ACCESS_TOKEN` - For API/CLI access
- `ID_TOKEN` - For web UI login
- `VIVARIA_DOCKER_GID` - Must match your Docker group GID
- `OPENAI_API_KEY` - OpenAI API key
- `ANTHROPIC_API_KEY` - Anthropic API key
- `AGENT_CPU_COUNT` - Default CPU per agent (1)
- `AGENT_RAM_GB` - Default RAM per agent (4)

### docker-compose.override.yml
Location: `vivaria/docker-compose.override.yml`

Key settings:
- `MP4_DOCKER_USE_GPUS: true` - Enable GPU support
- `NON_INTERVENTION_FULL_INTERNET_MODELS: .+` - Allow all models
- `SKIP_SAFETY_POLICY_CHECKING: true` - Skip safety checks
- GPU device reservations with `count: all`

### Caddyfile
Location: `vivaria/Caddyfile`

Must serve **HTTP** (not HTTPS) for Tailscale Serve to work:
```caddyfile
:4000 {
    handle /api/* {
        uri strip_prefix /api
        reverse_proxy http://server:4001
        encode gzip
    }
    handle {
        root * /srv
        file_server
        encode gzip
    }
}
```

## Architecture Overview

```
Your Laptop
    ↓ HTTPS (Tailscale)
Tailscale Network (encrypted tunnel)
    ↓
Tailscale Serve (on server)
    ├─ Port 4000: HTTPS → HTTP localhost:4000 → vivaria-ui-1 (Caddy)
    └─ Port 4001: HTTPS → HTTP localhost:4001 → vivaria-server-1

vivaria-server-1
    ├─ API endpoints
    ├─ Docker socket access (spawns agent containers)
    └─ GPU access (all 8 H100s)

vivaria-background-process-runner-1
    ├─ Processes queued tasks
    ├─ Docker socket access
    └─ GPU access (all 8 H100s)

vivaria-database-1
    └─ PostgreSQL (task state, runs, logs)
```

## Verification Checklist

Before running tasks, verify:

- [ ] All 4 containers are running and healthy: `sudo docker compose ps`
- [ ] GPU access works in server: `sudo docker exec vivaria-server-1 nvidia-smi -L`
- [ ] GPU access works in background runner: `sudo docker exec vivaria-background-process-runner-1 nvidia-smi -L`
- [ ] Docker socket access works: `sudo docker exec vivaria-server-1 docker ps`
- [ ] API responds: `curl http://localhost:4001/health`
- [ ] UI serves content: `curl http://localhost:4000`
- [ ] Tailscale Serve is configured: `sudo tailscale serve status`
- [ ] Web UI accessible from laptop: Visit `https://<hostname>.tailf03613.ts.net:4000`
- [ ] viv CLI configured correctly: `viv config list`

## Quick Reference Commands

```bash
# Check container status
sudo docker compose ps

# View container logs
sudo docker compose logs server --tail 50
sudo docker compose logs background-process-runner --tail 50

# Restart all services
sudo docker compose restart

# Fully restart (if GPU issues)
sudo docker compose down
sudo docker compose -f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.override.yml up --detach --wait

# Check Tailscale status
sudo tailscale status
sudo tailscale serve status

# Test GPU access
sudo docker exec vivaria-server-1 nvidia-smi
sudo docker exec vivaria-background-process-runner-1 nvidia-smi

# Test API
curl http://localhost:4001/health

# View environment config
cat vivaria/.env.server
```

## Access Information Summary

After setup, save this information:

- **Web UI**: `https://<your-hostname>.tailf03613.ts.net:4000`
- **API**: `https://<your-hostname>.tailf03613.ts.net:4001`
- **ID Token**: (from `.env.server`)
- **Access Token**: (from `.env.server`)

Configure viv CLI:
```bash
viv config set apiUrl https://<your-hostname>.tailf03613.ts.net:4001
viv config set uiUrl https://<your-hostname>.tailf03613.ts.net:4000
viv config set evalsToken <ACCESS_TOKEN>
```

## Next Steps

1. Access the web UI and verify you can log in
2. Run a simple test task to verify GPU access
3. Configure any additional task-specific environment variables in `deltamlbench/secrets.env`
4. Review agent resource limits in `.env.server` if needed (default 1 CPU, 4GB RAM per agent)

## Notes

- Tailscale Serve provides HTTPS automatically using Tailscale's built-in certificate management
- All traffic between your laptop and the server is encrypted through Tailscale's WireGuard tunnel
- The setup allows multiple users on the same Tailnet to access the Vivaria instance
- GPU resources are shared among all running tasks - adjust `AGENT_CPU_COUNT` and `AGENT_RAM_GB` if needed
