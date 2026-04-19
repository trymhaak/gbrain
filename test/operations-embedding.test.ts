import { afterEach, beforeEach, describe, expect, spyOn, test, mock } from 'bun:test';
import * as importFileModule from '../src/core/import-file.ts';
import * as linkExtractionModule from '../src/core/link-extraction.ts';
import { operationsByName } from '../src/core/operations.ts';

const originalOpenAI = process.env.OPENAI_API_KEY;
const originalVoyage = process.env.VOYAGE_API_KEY;

describe('put_page embedding provider detection', () => {
  beforeEach(() => {
    delete process.env.OPENAI_API_KEY;
    delete process.env.VOYAGE_API_KEY;
  });

  afterEach(() => {
    if (originalOpenAI) process.env.OPENAI_API_KEY = originalOpenAI;
    else delete process.env.OPENAI_API_KEY;

    if (originalVoyage) process.env.VOYAGE_API_KEY = originalVoyage;
    else delete process.env.VOYAGE_API_KEY;

    mock.restore();
  });

  test('keeps embeddings enabled for put_page when only VOYAGE_API_KEY is set', async () => {
    process.env.VOYAGE_API_KEY = 'voyage-key';

    const importSpy = spyOn(importFileModule, 'importFromContent').mockResolvedValue({
      slug: 'notes/voyage-only',
      status: 'imported',
      chunks: 0,
    } as any);
    spyOn(linkExtractionModule, 'isAutoLinkEnabled').mockResolvedValue(false);

    const putOp = operationsByName['put_page'];
    await putOp.handler(
      {
        dryRun: false,
        engine: {} as any,
        remote: false,
      } as any,
      {
        slug: 'notes/voyage-only',
        content: '---\ntype: note\ntitle: Voyage Only\n---\n\nHello.\n',
      },
    );

    expect(importSpy).toHaveBeenCalledTimes(1);
    expect(importSpy.mock.calls[0]?.[2]).toBe('---\ntype: note\ntitle: Voyage Only\n---\n\nHello.\n');
    expect(importSpy.mock.calls[0]?.[3]?.noEmbed).toBe(false);
  });
});
