# Redis Setup Guide

Seraph MCP uses an **in-memory cache by default** and requires zero configuration to get started. Redis is an **optional upgrade** that provides persistent, shared caching across multiple processes.

## When to Use Redis

Use Redis when you need:
- **Persistence**: Cache survives restarts
- **Shared cache**: Multiple MCP instances share the same cache
- **Higher capacity**: Better memory management and eviction policies
- **Production deployments**: Reliable caching for scaled environments

## Quick Start (Docker)

The easiest way to run Redis locally is using Docker Compose:

```bash
# Start Redis server
docker-compose up -d

# Check Redis is running
docker-compose ps

# View logs
docker-compose logs redis

# Stop Redis
docker-compose down

# Stop and remove all data
docker-compose down -v
```

This starts:
- **Redis server** on `localhost:6379`
- **RedisInsight UI** on `http://localhost:8001` (optional web management tool)

## Configure Seraph MCP to Use Redis

Once Redis is running, configure Seraph MCP by setting environment variables:

### Option 1: Environment Variables

```bash
# Set Redis URL (Seraph will auto-detect and use Redis)
export REDIS_URL=redis://localhost:6379/0

# Optional: explicitly set cache backend
export CACHE_BACKEND=redis

# Run Seraph MCP
fastmcp dev src/server.py
```

### Option 2: `.env` File

Create or edit `.env` in your project root:

```bash
# Redis Configuration
REDIS_URL=redis://localhost:6379/0
CACHE_BACKEND=redis  # Optional, auto-detected from REDIS_URL

# Optional: Redis connection tuning
REDIS_MAX_CONNECTIONS=10
REDIS_SOCKET_TIMEOUT=5

# Cache settings
CACHE_TTL_SECONDS=3600
CACHE_NAMESPACE=seraph
```

Then run:

```bash
fastmcp dev src/server.py
```

## Verify Redis Connection

Check that Seraph MCP is using Redis:

```bash
# Using the check_status tool
fastmcp dev src/server.py
# In another terminal:
# Call the check_status MCP tool and verify cache.backend=redis
```

Or check directly with Redis CLI:

```bash
# Install redis-cli if needed
# Ubuntu/Debian: apt-get install redis-tools
# macOS: brew install redis

# Ping Redis
redis-cli ping
# Should return: PONG

# List keys with seraph namespace
redis-cli --scan --pattern "seraph:*"
```

## Alternative: Install Redis Directly

If you don't want to use Docker:

### macOS
```bash
brew install redis
brew services start redis
```

### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install redis-server
sudo systemctl start redis
sudo systemctl enable redis
```

### Windows
Download and install from [Redis Windows releases](https://github.com/microsoftarchive/redis/releases) or use WSL.

## Redis Configuration Options

### Memory Limits
Configure Redis memory policy in `docker-compose.yml`:

```yaml
environment:
  - REDIS_ARGS=--maxmemory 512mb --maxmemory-policy allkeys-lru
```

### Password Protection
Add password authentication:

```yaml
environment:
  - REDIS_ARGS=--requirepass yourpassword
```

Then update your `.env`:

```bash
REDIS_URL=redis://:yourpassword@localhost:6379/0
```

### Persistence
Redis automatically saves data to disk. Data is stored in the `redis_data` Docker volume.

## Troubleshooting

### Connection Refused
```
Error: Connection refused at localhost:6379
```

**Solution**: Start Redis server
```bash
docker-compose up -d
# Or if using local Redis:
redis-server
```

### Redis Not Found
```
Error: redis_url is required when cache backend is 'redis'
```

**Solution**: Set `REDIS_URL` environment variable:
```bash
export REDIS_URL=redis://localhost:6379/0
```

### Permission Denied
```
Error: Could not connect to Redis at localhost:6379: Permission denied
```

**Solution**: Check Redis is running and accessible:
```bash
redis-cli ping
```

### Port Already in Use
```
Error: bind: address already in use
```

**Solution**: Change the port in `docker-compose.yml`:
```yaml
ports:
  - "6380:6379"  # Use port 6380 on host
```

Then update your `.env`:
```bash
REDIS_URL=redis://localhost:6380/0
```

## Production Deployment

For production, consider:

1. **Redis Cluster**: For high availability and horizontal scaling
2. **Managed Redis**: AWS ElastiCache, Redis Cloud, Azure Cache for Redis
3. **Monitoring**: Use RedisInsight or Prometheus exporters
4. **Backup**: Configure Redis persistence (AOF + RDB)
5. **Security**: Enable authentication, use TLS, restrict network access

Example production URL:
```bash
REDIS_URL=redis://:password@redis.production.example.com:6379/0
```

## Switching Back to Memory Cache

To switch back to the in-memory cache:

```bash
# Unset REDIS_URL
unset REDIS_URL

# Or set cache backend explicitly
export CACHE_BACKEND=memory

# Run Seraph MCP
fastmcp dev src/server.py
```

## Next Steps

- Explore RedisInsight UI at http://localhost:8001
- Monitor cache hit rates with `check_status` MCP tool
- Tune `CACHE_TTL_SECONDS` based on your workload

## Support

For issues or questions:
- Check [GitHub Issues](https://github.com/coderdayton/seraph-mcp/issues)
- Review [SDD.md](../SDD.md) for architecture details
- See [TESTING.md](../TESTING.md) for Redis testing guidelines
