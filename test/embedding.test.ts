import { describe, expect, test } from 'bun:test';

describe('embedding config', () => {
  test('defaults to OpenAI config when no embedding keys are present', async () => {
    const mod = await import('../src/core/embedding.ts');
    const config = mod.resolveEmbeddingConfig({});

    expect(config.provider).toBe('openai');
    expect(config.model).toBe('text-embedding-3-large');
    expect(config.enabled).toBe(false);
    expect(config.dimensions).toBe(1536);
  });

  test('enables OpenAI embeddings when OPENAI_API_KEY is present', async () => {
    const mod = await import('../src/core/embedding.ts');
    const config = mod.resolveEmbeddingConfig({ OPENAI_API_KEY: 'openai-key' });

    expect(config.provider).toBe('openai');
    expect(config.model).toBe('text-embedding-3-large');
    expect(config.enabled).toBe(true);
  });

  test('prefers Voyage when VOYAGE_API_KEY is present', async () => {
    const mod = await import('../src/core/embedding.ts');
    const config = mod.resolveEmbeddingConfig({
      OPENAI_API_KEY: 'openai-key',
      VOYAGE_API_KEY: 'voyage-key',
    });

    expect(config.provider).toBe('voyage');
    expect(config.model).toBe('voyage-large-2');
    expect(config.enabled).toBe(true);
    expect(config.dimensions).toBe(1536);
  });
});
