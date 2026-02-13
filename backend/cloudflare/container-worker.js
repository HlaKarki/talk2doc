import { Container, getContainer } from "@cloudflare/containers"
import { env as workerEnv } from "cloudflare:workers"

const REQUIRED_ENV_KEYS = [
  "DATABASE_URL",
  "OPENAI_API_KEY",
  "R2_ACCOUNT_ID",
  "R2_ACCESS_KEY_ID",
  "R2_SECRET_ACCESS_KEY",
]

function shouldRetryStartup(error) {
  const message = String(error?.message ?? error ?? "")
  return (
    message.includes("not listening") ||
    message.includes("not running") ||
    message.includes("consider calling start()")
  )
}

export class Talk2DocBackendContainer extends Container {
  defaultPort = 8000
  sleepAfter = "15m"
  enableInternet = true
  pingEndpoint = "/health"
  // Pass Worker env/secrets into container process env.
  envVars = {
    DATABASE_URL: workerEnv.DATABASE_URL,
    OPENAI_API_KEY: workerEnv.OPENAI_API_KEY,
    EMBEDDING_MODEL: workerEnv.EMBEDDING_MODEL ?? "text-embedding-3-small",
    EMBEDDING_DIMENSIONS: String(workerEnv.EMBEDDING_DIMENSIONS ?? "1536"),
    CHUNK_SIZE: String(workerEnv.CHUNK_SIZE ?? "1000"),
    CHUNK_OVERLAP: String(workerEnv.CHUNK_OVERLAP ?? "200"),
    R2_ACCOUNT_ID: workerEnv.R2_ACCOUNT_ID,
    R2_ACCESS_KEY_ID: workerEnv.R2_ACCESS_KEY_ID,
    R2_SECRET_ACCESS_KEY: workerEnv.R2_SECRET_ACCESS_KEY,
    R2_BUCKET_NAME: workerEnv.R2_BUCKET_NAME ?? "talk2doc",
    CORS_ORIGINS: workerEnv.CORS_ORIGINS ?? "http://localhost:3000",
    AUTO_INIT_DB: String(workerEnv.AUTO_INIT_DB ?? "false"),
  }
}

export default {
  async fetch(request, env) {
    const missing = REQUIRED_ENV_KEYS.filter((key) => {
      const value = env[key]
      return typeof value !== "string" || value.length === 0
    })

    if (missing.length > 0) {
      return Response.json(
        {
          error: "Missing required Worker secrets/vars for backend container startup.",
          missing,
        },
        { status: 500 }
      )
    }

    // Use a stable named container for API traffic to avoid random cold instance selection.
    const container = await getContainer(env.TALK2DOC_BACKEND, "api-default")
    if (typeof container.start === "function") {
      await container.start()
    }

    try {
      return await container.fetch(request)
    } catch (error) {
      if (!shouldRetryStartup(error)) {
        throw error
      }

      // Cold-start guard: retry once after nudging start.
      if (typeof container.start === "function") {
        await container.start()
      }
      return container.fetch(request)
    }
  },
}
