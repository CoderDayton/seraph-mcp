# Seraph MCP — Docker Setup

This directory contains Docker configurations for running Seraph MCP infrastructure components.

## Quick Start — Redis Only (Recommended)

Most users only need Redis for persistent caching. The Seraph MCP server itself runs directly on your host machine.

```bash
# From the docker/ directory
docker-compose up -d

# Or from project root
docker-compose -f docker/docker-compose.yml up -d
```

This starts:
- **Redis server** on `localhost:6379`
- **RedisInsight UI** at `http://localhost:8001`

Then configure Seraph MCP to use Redis:
```bash
export REDIS_URL=redis://localhost:6379/0
```

Stop Redis:
```bash
docker-compose down
```

## Development Setup

Use the development compose file for a more permissive Redis configuration:

```bash
docker-compose -f docker-compose.dev.yml up -d
```

This includes:
- Redis with 1GB memory limit (vs 512MB in production)
- No password requirement
- Combined RedisInsight in single container

## Full Application Container (Optional)

Most users don't need this — run Seraph MCP directly with `uvx seraph-mcp` instead.

If you want to containerize the entire application:

```bash
# Build from project root
docker build -f docker/Dockerfile -t seraph-mcp:latest .

# Run (stdio mode)
docker run --rm -i seraph-mcp:latest
```

## Files

- **`docker-compose.yml`** — Production Redis setup with RedisInsight
- **`docker-compose.dev.yml`** — Development Redis setup (simplified)
- **`Dockerfile`** — Full application container (optional, rarely needed)
- **`.dockerignore`** — Files to exclude from Docker builds

## Why Docker?

**Redis Only:**
- Zero-dependency Redis server for local development
- RedisInsight web UI for cache inspection and debugging
- Persistent storage that survives restarts

**Full Container:**
- Rarely needed — most users run `uvx seraph-mcp` directly
- Useful for:
  - Deployment to container platforms
  - Isolated testing environments
  - CI/CD pipelines

## Configuration

### Redis Connection

The default Redis URL is `redis://localhost:6379/0`. To use a different database or add authentication:

```bash
# Different database number
export REDIS_URL=redis://localhost:6379/5

# With password
export REDIS_URL=redis://:yourpassword@localhost:6379/0

# Remote Redis
export REDIS_URL=redis://your-redis-host:6379/0
```

### Memory Limits

Edit `docker-compose.yml` to adjust Redis memory:

```yaml
environment:
  - REDIS_ARGS=--maxmemory 1gb --maxmemory-policy allkeys-lru
```

### Persistence

Redis data is stored in Docker volumes:
- Production: `redis_data`
- Development: `redis_dev_data`

Remove volumes to clear all data:
```bash
docker-compose down -v
```

## Ports

| Service | Port | Purpose |
|---------|------|---------|
| Redis | 6379 | Redis server (TCP) |
| RedisInsight | 8001 | Web UI for Redis management |

## Troubleshooting

### Port Already in Use

If port 6379 is already in use:

```bash
# Check what's using the port
lsof -i :6379

# Kill the process or change the port in docker-compose.yml:
ports:
  - "6380:6379"  # Map to different host port

# Then update your REDIS_URL
export REDIS_URL=redis://localhost:6380/0
```

### Connection Refused

Ensure Redis is running:
```bash
docker-compose ps
```

Test connection:
```bash
redis-cli -h localhost -p 6379 ping
# Expected: PONG
```

### Clear Everything

Remove all containers, volumes, and networks:
```bash
docker-compose down -v
docker system prune -a
```

## Health Checks

Both compose files include health checks for Redis:

```bash
# Check Redis health
docker-compose ps

# View logs
docker-compose logs redis

# Access Redis CLI
docker exec -it seraph-redis redis-cli
```

## Production Deployment

For production deployment, consider:

1. **Use managed Redis** (AWS ElastiCache, Redis Cloud, etc.) instead of self-hosted
2. **Enable authentication** with strong passwords
3. **Configure TLS/SSL** for encrypted connections
4. **Set memory limits** appropriate for your workload
5. **Enable persistence** (RDB/AOF) for data durability

See the main [README](../README.md) for more production configuration options.
