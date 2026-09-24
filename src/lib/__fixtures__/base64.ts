/** Decode a base64 fixture (the GGUF header fixtures are stored as .gguf.b64 text). */
export function fromBase64(b64: string): ArrayBuffer {
  const bin = atob(b64.trim());
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out.buffer;
}
