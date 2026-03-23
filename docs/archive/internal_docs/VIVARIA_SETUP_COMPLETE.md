# Vivaria Setup Complete ✓

## Quick Start Commands

**Start Vivaria (with GPU support):**
```bash
cd /home/ubuntu/coding-agent/Coding-Agent-For-REBench/vivaria
sudo docker compose -f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.override.yml --env-file .env.server up -d
```

**Check Status:**
```bash
cd /home/ubuntu/coding-agent/Coding-Agent-For-REBench/vivaria
sudo docker compose -f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.override.yml --env-file .env.server ps
```

**Access Web UI:** http://localhost:4000

**Run a Task:**
```bash
cd /home/ubuntu/coding-agent/Coding-Agent-For-REBench
source ~/.bashrc_pwc  # Activate pwc environment
./vivaria_manager.sh start pwc_mnist_gatedgcn
```

---

## What's Been Installed

1. **Miniconda3** - Installed at `~/miniconda3`
2. **PWC Conda Environment** - Python 3.11.13 environment
3. **Vivaria CLI (viv)** - Installed in the pwc environment
4. **Vivaria Services** - Running via Docker Compose with GPU support

## Service Status

All Vivaria Docker containers are running and healthy:
- **Database** - PostgreSQL on port 5432
- **Server** - API server on port 4001
- **UI** - Web interface on port 4000
- **Background Process Runner** - For task execution

## How to Use

### Activate PWC Environment
```bash
source ~/.bashrc_pwc
```

Or manually:
```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pwc
```

### Using the VIV CLI
Once in the pwc environment:
```bash
viv --help                  # Show all commands
viv config list             # View configuration
viv task --help             # Task management
viv run --help              # Run agents on tasks
```

### Using the Vivaria Manager Script
The project includes a comprehensive manager script:
```bash
cd /home/ubuntu/coding-agent/Coding-Agent-For-REBench
./vivaria_manager.sh help
```

Common commands:
```bash
./vivaria_manager.sh tasks              # List all available tasks
./vivaria_manager.sh status             # Show status of all tasks
./vivaria_manager.sh vivaria-status     # Check Vivaria services
./vivaria_manager.sh start <task>       # Start a specific task
```

### Managing Docker Containers

**Note:** You need to either:
1. Use `sudo` before docker commands, OR
2. Log out and log back in for docker group permissions to take effect

**Important:** Vivaria must be started with GPU support using all three compose files:
```bash
cd /home/ubuntu/coding-agent/Coding-Agent-For-REBench/vivaria
```

Common commands (from the vivaria directory):
```bash
# Check status
sudo docker compose -f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.override.yml --env-file .env.server ps

# Start services (with GPU support)
sudo docker compose -f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.override.yml --env-file .env.server up -d

# Stop services
sudo docker compose -f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.override.yml --env-file .env.server down

# View logs
sudo docker compose -f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.override.yml --env-file .env.server logs -f

# Restart a specific service
sudo docker compose -f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.override.yml --env-file .env.server restart server
```

**Shorter alias for convenience:**
```bash
alias viv-compose='sudo docker compose -f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.override.yml --env-file .env.server'
# Then use: viv-compose ps, viv-compose up -d, etc.
```

## Configuration Files

- **Vivaria config**: `~/.config/viv-cli/config.json`
- **Vivaria env files**: `/home/ubuntu/coding-agent/Coding-Agent-For-REBench/vivaria/.env.server` and `.env.db`
- **Docker override**: `/home/ubuntu/coding-agent/Coding-Agent-For-REBench/vivaria/docker-compose.override.yml`
- **Task secrets**: `/home/ubuntu/coding-agent/Coding-Agent-For-REBench/deltamlbench/secrets.env`

## API Endpoints

- **API**: http://localhost:4001
- **Web UI**: http://localhost:4000
- **Health Check**: http://localhost:4001/health

## Testing the Setup

Test API connectivity:
```bash
curl http://localhost:4001/health
```

Expected response: `{"result":{"data":"ok"}}`

## Running a PWC Task

Example workflow to run a task:
```bash
# 1. Activate environment
source ~/.bashrc_pwc

# 2. Use the manager script
cd /home/ubuntu/coding-agent/Coding-Agent-For-REBench
./vivaria_manager.sh start pwc_mnist_gatedgcn

# 3. Monitor the task
./vivaria_manager.sh status pwc_mnist_gatedgcn
./vivaria_manager.sh monitor pwc_mnist_gatedgcn
```

Or use viv directly:
```bash
viv run \
    --agent-path modular-public \
    --task-family-path deltamlbench/pwc_mnist_gatedgcn \
    --env-file-path deltamlbench/secrets.env \
    --max_tokens 10000000 \
    --max_actions 3000 \
    pwc_mnist_gatedgcn/main
```

## Available PWC Tasks

47 PWC tasks are available in RE-Bench. See the full list:
```bash
./vivaria_manager.sh tasks | grep pwc_
```

## Troubleshooting

### Docker Permission Issues
If you get "permission denied" errors with docker:
```bash
# Either use sudo:
sudo docker ps

# Or log out and log back in to refresh group membership
```

### Vivaria Services Not Running
```bash
cd /home/ubuntu/coding-agent/Coding-Agent-For-REBench/vivaria
sudo docker compose -f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.override.yml --env-file .env.server up -d
```

### Services Show as "Unhealthy"
If services show as unhealthy, make sure you're using the GPU compose file:
```bash
cd /home/ubuntu/coding-agent/Coding-Agent-For-REBench/vivaria
sudo docker compose -f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.override.yml --env-file .env.server down
sudo docker compose -f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.override.yml --env-file .env.server up -d
```

Verify GPU access inside containers:
```bash
sudo docker exec vivaria-server-1 nvidia-smi
```

### Check Container Logs
```bash
cd /home/ubuntu/coding-agent/Coding-Agent-For-REBench/vivaria
sudo docker compose -f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.override.yml --env-file .env.server logs server
sudo docker compose -f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.override.yml --env-file .env.server logs background-process-runner
```

### "Unable to transform response from server" Error in UI
This usually means:
1. Services aren't running with GPU support (use the full compose command above)
2. API keys aren't configured (they should be in `/home/ubuntu/coding-agent/Coding-Agent-For-REBench/vivaria/.env.server`)
3. Services need to be fully recreated with `down` then `up -d`

### "You don't have permission to use model" Error
If you get an error like: `You don't have permission to use model "claude-sonnet-4-20250514"`:

1. **Check which API keys are active** in the running container:
   ```bash
   docker exec vivaria-server-1 printenv | grep -E "OPENAI_API_KEY|GEMINI_API_KEY|ANTHROPIC_API_KEY"
   ```
   
   **If `OPENAI_API_KEY` is set, only OpenAI models are permitted!**
   
   To allow Claude/Gemini models, ensure **only one of these is set:**
   - `ANTHROPIC_API_KEY` (allows all models, including Anthropic and Gemini)
   - `GEMINI_API_KEY` (allows all models, including Gemini and Anthropic)
   
   **Do NOT set `OPENAI_API_KEY`** if you want to use non-OpenAI models.

2. **Edit `.env.server`** to comment out OpenAI:
   ```bash
   # In /home/ubuntu/coding-agent/Coding-Agent-For-REBench/vivaria/.env.server:
   ANTHROPIC_API_KEY=...  # KEEP this
   # OPENAI_API_KEY=...   # COMMENT this out
   ```

3. **Restart services** after changing `.env.server`:
   ```bash
   cd /home/ubuntu/coding-agent/Coding-Agent-For-REBench/vivaria
   sudo docker compose -f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.override.yml --env-file .env.server down
   sudo docker compose -f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.override.yml --env-file .env.server up -d
   ```

4. **Important**: Any running agent tasks must be restarted for the new configuration to take effect.

## Next Steps

1. Log out and log back in to enable docker commands without sudo
2. Review the task documentation in `/home/ubuntu/coding-agent/Coding-Agent-For-REBench/deltamlbench/`
3. Ensure secrets are configured in `deltamlbench/secrets.env` (API keys, tokens, etc.)
4. Start running PWC tasks!

## GPU Support

Vivaria is configured with GPU support:
- **GPU Model**: NVIDIA A100-SXM4-40GB
- **CUDA Version**: 12.8
- **Driver Version**: 570.148.08

### Key Configuration Files

The GPU support requires three compose files to be loaded together:
1. `docker-compose.yml` - Base configuration
2. `docker-compose.gpu.yml` - GPU-specific settings and healthchecks
3. `docker-compose.override.yml` - Local overrides (HTTP mode, GPU reservations)

### API Keys Configuration

The following API keys are configured in `.env.server`:
- `OPENAI_API_KEY` - For GPT models
- `ANTHROPIC_API_KEY` - For Claude models
- `GOOGLE_API_KEY` - For Gemini models

These must be present for Vivaria to run agents successfully.

**Important: Model Permissions and Limitations**

Vivaria's `builtin` middleman has **hardcoded provider priority** in its source code:
1. OPENAI_API_KEY (checked first)
2. GEMINI_API_KEY (checked second)  
3. ANTHROPIC_API_KEY (checked last)

**The order in `.env.server` doesn't matter** - the code always checks in this order!

**How it affects model permissions:**
- **If `OPENAI_API_KEY` is set** → Only OpenAI models permitted (restrictive)
- **If only `ANTHROPIC_API_KEY` is set** → **ALL models from ALL providers permitted** (permissive)
- **If only `GEMINI_API_KEY` is set** → ALL models from ALL providers permitted (permissive)

**Current Configuration (allows Claude models):**

In `.env.server`, **only** `ANTHROPIC_API_KEY` is active:
```bash
ANTHROPIC_API_KEY=sk-ant-...  # ACTIVE - allows all models

# These are commented out:
# GEMINI_API_KEY=...
# OPENAI_API_KEY=...
```

**Limitation:** Agents can use Claude and Gemini models, but **NOT OpenAI models** (no API key available).

**To use all three providers simultaneously:**
- Requires using a remote middleman service with custom permission logic
- OR patching Vivaria's source code to change provider priority
- The builtin middleman cannot support all three providers at once

After changing `.env.server`, restart services:
```bash
cd /home/ubuntu/coding-agent/Coding-Agent-For-REBench/vivaria
sudo docker compose -f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.override.yml --env-file .env.server down
sudo docker compose -f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.override.yml --env-file .env.server up -d
```

### VIV CLI Configuration

The viv CLI is configured to use HTTP (not HTTPS) for local development:
- `apiUrl`: http://localhost:4001
- `uiUrl`: http://localhost:4000

This configuration is stored in `~/.config/viv-cli/config.json`.

