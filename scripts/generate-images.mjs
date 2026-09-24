#!/usr/bin/env node
// Generates the PWA icons (public/icon-192.png, public/icon-512.png) and the Open Graph /
// Twitter card image (public/og.png) with a tiny hand-rolled PNG encoder — Node built-ins only
// (zlib for DEFLATE + CRC32), no image library. Re-run after changing the brand mark:
//   node scripts/generate-images.mjs
//
// The mark is the same three-bar glyph as public/favicon.svg (bars at x=7/13.5/20, y=18/12/6,
// height 8/14/20 in a 32x32 box), rasterized at each target size instead of traced from the SVG.

import { deflateSync, crc32 } from 'node:zlib';
import { writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const here = path.dirname(fileURLToPath(import.meta.url));
const publicDir = path.join(here, '..', 'public');

const BG = [0x11, 0x14, 0x18, 255]; // --surface-2-ish dark background, matches favicon.svg
const GREEN = [0x3e, 0xcf, 0x7a, 255]; // --ok
const BLUE = [0x6a, 0xa8, 0xff, 255]; // --accent

/** PNG chunk: 4-byte length + 4-byte type + data + 4-byte CRC32 of (type + data). */
function chunk(type, data) {
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length, 0);
  const typeBuf = Buffer.from(type, 'ascii');
  const crcBuf = Buffer.alloc(4);
  crcBuf.writeUInt32BE(crc32(Buffer.concat([typeBuf, data])) >>> 0, 0);
  return Buffer.concat([len, typeBuf, data, crcBuf]);
}

/** Encodes an 8-bit RGBA pixel buffer (width*height*4 bytes, row-major) as a PNG. */
function encodePng(width, height, rgba) {
  const signature = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
  const ihdrData = Buffer.alloc(13);
  ihdrData.writeUInt32BE(width, 0);
  ihdrData.writeUInt32BE(height, 4);
  ihdrData[8] = 8; // bit depth
  ihdrData[9] = 6; // color type: RGBA
  // bytes 10-12 (compression, filter, interlace) default to 0
  const ihdr = chunk('IHDR', ihdrData);

  // One "none" filter byte (0) per scanline, per the PNG spec.
  const stride = width * 4;
  const raw = Buffer.alloc((stride + 1) * height);
  for (let y = 0; y < height; y++) {
    raw[y * (stride + 1)] = 0;
    rgba.copy(raw, y * (stride + 1) + 1, y * stride, y * stride + stride);
  }
  const idat = chunk('IDAT', deflateSync(raw));
  const iend = chunk('IEND', Buffer.alloc(0));
  return Buffer.concat([signature, ihdr, idat, iend]);
}

function fill(rgba, pixelCount, color) {
  for (let i = 0; i < pixelCount; i++) rgba.set(color, i * 4);
}

function rect(rgba, canvasWidth, canvasHeight, x, y, w, h, color) {
  const x0 = Math.max(0, Math.round(x));
  const y0 = Math.max(0, Math.round(y));
  const x1 = Math.min(canvasWidth, Math.round(x + w));
  const y1 = Math.min(canvasHeight, Math.round(y + h));
  for (let yy = y0; yy < y1; yy++) {
    for (let xx = x0; xx < x1; xx++) {
      const idx = (yy * canvasWidth + xx) * 4;
      rgba[idx] = color[0];
      rgba[idx + 1] = color[1];
      rgba[idx + 2] = color[2];
      rgba[idx + 3] = color[3];
    }
  }
}

/** The three-bar mark, scaled from its 32x32 favicon coordinates, drawn at (originX, originY). */
function drawMark(rgba, canvasWidth, canvasHeight, originX, originY, scale) {
  const bars = [
    { x: 7, y: 18, w: 5, h: 8, color: GREEN },
    { x: 13.5, y: 12, w: 5, h: 14, color: GREEN },
    { x: 20, y: 6, w: 5, h: 20, color: BLUE },
  ];
  for (const bar of bars) {
    rect(rgba, canvasWidth, canvasHeight, originX + bar.x * scale, originY + bar.y * scale, bar.w * scale, bar.h * scale, bar.color);
  }
}

/** Square app icon: the mark filling most of the square, on the brand background. */
function makeIcon(size) {
  const rgba = Buffer.alloc(size * size * 4);
  fill(rgba, size * size, BG);
  const scale = size / 32;
  drawMark(rgba, size, size, 0, 0, scale);
  return encodePng(size, size, rgba);
}

/** Generic 1200x630 Open Graph / Twitter card: the mark centered on the brand background. */
function makeOg(width, height) {
  const rgba = Buffer.alloc(width * height * 4);
  fill(rgba, width * height, BG);
  const markSize = 320; // rendered footprint of the 32-unit mark at this scale
  const scale = markSize / 32;
  drawMark(rgba, width, height, (width - markSize) / 2, (height - markSize) / 2, scale);
  return encodePng(width, height, rgba);
}

writeFileSync(path.join(publicDir, 'icon-192.png'), makeIcon(192));
writeFileSync(path.join(publicDir, 'icon-512.png'), makeIcon(512));
writeFileSync(path.join(publicDir, 'og.png'), makeOg(1200, 630));
console.log('Wrote public/icon-192.png, public/icon-512.png, public/og.png');
