# Vivaria Tailscale Access Instructions

## Access URLs via Tailscale

Your Vivaria instance is now accessible from your laptop via Tailscale at:

- **Web UI**: http://100.89.138.94:4000 or http://132-145-140-184:4000
- **API**: http://100.89.138.94:4001 or http://132-145-140-184:4001
- **Database** (PostgreSQL): 100.89.138.94:5432

## Setting up Vivaria CLI on Your Laptop

### 1. Install Vivaria CLI on your laptop

```bash
# Option A: Using pip
pip install viv-cli

# Option B: From source
git clone https://github.com/METR/vivaria.git
cd vivaria/cli
pip install -e .
```

### 2. Configure Vivaria CLI to use Tailscale endpoint

Create or edit `~/.config/viv-cli/config.json` on your laptop:

```json
{
  "apiUrl": "http://100.89.138.94:4001",
  "uiUrl": "http://100.89.138.94:4000",
  "evalsToken": ""
}
```

Or use the CLI commands:
```bash
viv config set apiUrl http://100.89.138.94:4001
viv config set uiUrl http://100.89.138.94:4000
viv config set evalsToken ""
```

### 3. Test the connection

```bash
# Test API connectivity
curl http://100.89.138.94:4001/health

# Or use viv CLI
viv --version
viv config list
```

## Running Tasks from Your Laptop

Once configured, you can run tasks from your laptop:

```bash
# Example: Run a task
viv run \
  --agent-path modular-public \
  --task-family-path deltamlbench/pwc_cat2000_sum \
  --max_tokens 100000 \
  --max_actions 100 \
  pwc_cat2000_sum/main
```

## Security Notes

1. **Current Setup**: The services are accessible on all network interfaces (0.0.0.0)
2. **Tailscale Security**: Access is restricted to devices on your Tailscale network
3. **API Keys**: Your Anthropic and OpenAI keys are configured on the server

## Fixing Auth0 Error (Secure Origin Required)

The UI shows an Auth0 error because it's being accessed over HTTP (not HTTPS). Auth0 requires a secure origin. You have three options:

### Option 1: Use SSH Port Forwarding (Recommended)
This creates a localhost tunnel, which Auth0 considers secure:

```bash
# From your laptop, create SSH tunnel
ssh -L 4000:localhost:4000 -L 4001:localhost:4001 ubuntu@100.89.138.94

# Keep this terminal open, then in a new terminal/browser:
# Access UI at: http://localhost:4000
# Access API at: http://localhost:4001
```

### Option 2: Use Tailscale Serve (HTTPS)
Tailscale can provide HTTPS certificates:

```bash
# On the VM, expose the service via Tailscale with HTTPS
sudo tailscale serve https:4000 / http://localhost:4000
sudo tailscale serve https:4001 / http://localhost:4001

# Then access via:
# https://132-145-140-184.tailnet-name.ts.net:4000
```

### Option 3: Disable Auth0 Check (Development Only)
We can bypass the Auth0 initialization, but this requires modifying the UI code.

## Troubleshooting

### If you see a blank white screen in the UI:

This happens when the UI container hasn't picked up the updated environment variables. Fix it by:

```bash
# On the VM, recreate the UI container
cd /home/ubuntu/Coding-Agent-For-REBench/vivaria
sudo docker compose -f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.override.yml --env-file .env.server down ui
sudo docker compose -f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.override.yml --env-file .env.server up -d ui
```

### If you can't connect:

1. **Check Tailscale is connected on both devices**:
   ```bash
   tailscale status
   ```

2. **Test basic connectivity**:
   ```bash
   ping 100.89.138.94
   ```

3. **Check if ports are accessible**:
   ```bash
   nc -zv 100.89.138.94 4001
   ```

4. **Verify services are running on the VM**:
   ```bash
   ssh user@100.89.138.94
   sudo docker ps | grep vivaria
   ```

## Alternative: SSH Port Forwarding

If Tailscale direct access doesn't work, you can use SSH port forwarding:

```bash
# From your laptop
ssh -L 4000:localhost:4000 -L 4001:localhost:4001 user@100.89.138.94

# Then access locally
http://localhost:4000  # UI
http://localhost:4001  # API
```