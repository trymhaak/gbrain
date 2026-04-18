import { describe, test, expect, mock } from 'bun:test';
import type { BrainEngine } from '../src/core/engine.ts';

mock.module('../src/core/embedding.ts', () => ({
  EMBEDDING_MODEL: 'voyage-large-2',
  embedBatch: async (texts: string[]) => texts.map(() => new Float32Array([1, 2, 3])),
}));

const { importFromContent } = await import('../src/core/import-file.ts');

function mockEngine(overrides: Partial<Record<string, any>> = {}): BrainEngine {
  const calls: { method: string; args: any[] }[] = [];
  const track = (method: string) => (...args: any[]) => {
    calls.push({ method, args });
    if (overrides[method]) return overrides[method](...args);
    return Promise.resolve(null);
  };

  const engine = new Proxy({} as any, {
    get(_, prop: string) {
      if (prop === '_calls') return calls;
      if (prop === 'getTags') return overrides.getTags || (() => Promise.resolve([]));
      if (prop === 'getPage') return overrides.getPage || (() => Promise.resolve(null));
      if (prop === 'transaction') return async (fn: (tx: BrainEngine) => Promise<any>) => fn(engine);
      return track(prop);
    },
  });
  return engine;
}

describe('importFromContent embedding metadata', () => {
  test('stores EMBEDDING_MODEL on chunks when embeddings are generated', async () => {
    const engine = mockEngine();

    const result = await importFromContent(
      engine,
      'concepts/with-embedding-model',
      `---\ntype: concept\ntitle: Embedded\n---\n\nCompiled truth text.\n\n---\n\n- 2026-01-01: Timeline item.\n`,
    );

    expect(result.status).toBe('imported');

    const calls = (engine as any)._calls;
    const chunkCall = calls.find((c: any) => c.method === 'upsertChunks');
    expect(chunkCall).toBeTruthy();

    const chunks = chunkCall.args[1];
    expect(chunks.length).toBeGreaterThan(0);
    for (const chunk of chunks) {
      expect(chunk.embedding).toBeInstanceOf(Float32Array);
      expect(chunk.model).toBe('voyage-large-2');
    }
  });
});
