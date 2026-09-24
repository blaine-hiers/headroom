// Minimal ESM loader so scripts/*.mjs can `import` this repo's extensionless,
// bundler-style TS sources (src/lib/**) directly under Node's native TS
// type-stripping, without adding a bundler dependency (tsx, ts-node, ...).
// Node strips types on `.ts` files but does not add ".ts" for extensionless
// relative specifiers (that's a bundler-only convention) — this hook does that
// one thing and nothing else.
import { existsSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, resolve as resolvePath } from 'node:path';

export async function resolve(specifier, context, nextResolve) {
  if (specifier.startsWith('.') && !/\.[a-zA-Z0-9]+$/.test(specifier)) {
    const baseDir = dirname(fileURLToPath(context.parentURL));
    const candidate = resolvePath(baseDir, `${specifier}.ts`);
    if (existsSync(candidate)) {
      return nextResolve(pathToFileURL(candidate).href, context);
    }
  }
  return nextResolve(specifier, context);
}
