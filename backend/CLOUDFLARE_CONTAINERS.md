# Deploy Backend To Cloudflare Containers

This backend is set up for Cloudflare Containers with:

- Docker image: `backend/Dockerfile`
- Worker bridge: `backend/cloudflare/container-worker.js`
- Wrangler config: `backend/wrangler.containers.jsonc`

## 1) Install JS dependency (once)

From repo root:

```bash
npm install -D @cloudflare/containers
```

## 2) Authenticate Wrangler

```bash
npx wrangler whoami
```

If not logged in:

```bash
npx wrangler login
```

## 3) Set required Worker secrets

Secrets required for container startup:

```bash
npx wrangler secret put DATABASE_URL -c backend/wrangler.containers.jsonc
npx wrangler secret put OPENAI_API_KEY -c backend/wrangler.containers.jsonc
npx wrangler secret put R2_ACCOUNT_ID -c backend/wrangler.containers.jsonc
npx wrangler secret put R2_ACCESS_KEY_ID -c backend/wrangler.containers.jsonc
npx wrangler secret put R2_SECRET_ACCESS_KEY -c backend/wrangler.containers.jsonc
```

Optional runtime vars can be changed in `backend/wrangler.containers.jsonc` under `vars`:

- `R2_BUCKET_NAME`
- `CORS_ORIGINS`
- `EMBEDDING_MODEL`
- `EMBEDDING_DIMENSIONS`
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- `AUTO_INIT_DB` (recommended `false` in container runtime)

## 4) Deploy

```bash
npx wrangler deploy -c backend/wrangler.containers.jsonc
```

Cloudflare container deployments are rolling, not instant.

## 5) Verify

Use the URL returned by deploy:

```bash
curl https://<your-worker-url>/health
```

Expected:

```json
{"status":"healthy"}
```

## Notes

- Container instances are ephemeral. Persistent state should stay in Postgres/R2.
- The Worker bridge uses `getRandom()` for stateless API load distribution.
- CORS is now environment-driven via `CORS_ORIGINS` (comma-separated list).
