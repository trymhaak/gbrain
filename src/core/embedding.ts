/**
 * Embedding Service
 * Supports OpenAI by default, and Voyage when VOYAGE_API_KEY is present.
 *
 * Current storage schema is vector(1536), so Voyage defaults to voyage-large-2
 * to avoid a database migration during local bootstrap.
 * Retry with exponential backoff (4s base, 120s cap, 5 retries).
 * 8000 character input truncation.
 */

import OpenAI from 'openai';

type EmbeddingProvider = 'openai' | 'voyage';

interface EmbeddingConfig {
  provider: EmbeddingProvider;
  model: string;
  dimensions: number;
  enabled: boolean;
}

const DIMENSIONS = 1536;

export function resolveEmbeddingConfig(env: NodeJS.ProcessEnv | Record<string, string | undefined> = process.env): EmbeddingConfig {
  const provider: EmbeddingProvider = env.VOYAGE_API_KEY ? 'voyage' : 'openai';

  return {
    provider,
    model: provider === 'voyage' ? 'voyage-large-2' : 'text-embedding-3-large',
    dimensions: DIMENSIONS,
    enabled: Boolean(env.VOYAGE_API_KEY || env.OPENAI_API_KEY),
  };
}

const CONFIG = resolveEmbeddingConfig();
const PROVIDER = CONFIG.provider;
const MODEL = CONFIG.model;
const MAX_CHARS = 8000;
const MAX_RETRIES = 5;
const BASE_DELAY_MS = 4000;
const MAX_DELAY_MS = 120000;
const BATCH_SIZE = 100;

let client: OpenAI | null = null;

function getClient(): OpenAI {
  if (!client) {
    client = new OpenAI(PROVIDER === 'voyage'
      ? {
          apiKey: process.env.VOYAGE_API_KEY,
          baseURL: 'https://api.voyageai.com/v1',
        }
      : undefined);
  }
  return client;
}

export function embeddingsEnabled(): boolean {
  return resolveEmbeddingConfig().enabled;
}

export async function embed(text: string): Promise<Float32Array> {
  const truncated = text.slice(0, MAX_CHARS);
  const result = await embedBatch([truncated]);
  return result[0];
}

export async function embedBatch(texts: string[]): Promise<Float32Array[]> {
  const truncated = texts.map(t => t.slice(0, MAX_CHARS));
  const results: Float32Array[] = [];

  // Process in batches of BATCH_SIZE
  for (let i = 0; i < truncated.length; i += BATCH_SIZE) {
    const batch = truncated.slice(i, i + BATCH_SIZE);
    const batchResults = await embedBatchWithRetry(batch);
    results.push(...batchResults);
  }

  return results;
}

async function embedBatchWithRetry(texts: string[]): Promise<Float32Array[]> {
  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    try {
      const response = await getClient().embeddings.create(
        PROVIDER === 'voyage'
          ? {
              model: MODEL,
              input: texts,
            }
          : {
              model: MODEL,
              input: texts,
              dimensions: DIMENSIONS,
            },
      );

      // Sort by index to maintain order
      const sorted = response.data.sort((a, b) => a.index - b.index);
      return sorted.map(d => new Float32Array(d.embedding));
    } catch (e: unknown) {
      if (attempt === MAX_RETRIES - 1) throw e;

      // Check for rate limit with Retry-After header
      let delay = exponentialDelay(attempt);

      if (e instanceof OpenAI.APIError && e.status === 429) {
        const retryAfter = e.headers?.['retry-after'];
        if (retryAfter) {
          const parsed = parseInt(retryAfter, 10);
          if (!isNaN(parsed)) {
            delay = parsed * 1000;
          }
        }
      }

      await sleep(delay);
    }
  }

  // Should not reach here
  throw new Error('Embedding failed after all retries');
}

function exponentialDelay(attempt: number): number {
  const delay = BASE_DELAY_MS * Math.pow(2, attempt);
  return Math.min(delay, MAX_DELAY_MS);
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

export { MODEL as EMBEDDING_MODEL, DIMENSIONS as EMBEDDING_DIMENSIONS };
