import { describe, expect, it } from "vitest";
import { crc32, createZip, uniqueNames } from "./zip";

const bytes = (text: string) => new TextEncoder().encode(text);

/** Read the entries back the way an unzip tool would (via the central directory). */
function readZip(buffer: ArrayBuffer): Array<{ name: string; data: string; crc: number }> {
  const view = new DataView(buffer);
  const end = buffer.byteLength - 22;
  expect(view.getUint32(end, true)).toBe(0x06054b50);
  const count = view.getUint16(end + 10, true);
  let pos = view.getUint32(end + 16, true);
  const decoder = new TextDecoder();
  const entries = [];
  for (let i = 0; i < count; i++) {
    expect(view.getUint32(pos, true)).toBe(0x02014b50);
    const crc = view.getUint32(pos + 16, true);
    const size = view.getUint32(pos + 20, true);
    const nameLength = view.getUint16(pos + 28, true);
    const localOffset = view.getUint32(pos + 42, true);
    const name = decoder.decode(new Uint8Array(buffer, pos + 46, nameLength));
    expect(view.getUint32(localOffset, true)).toBe(0x04034b50);
    const dataStart = localOffset + 30 + view.getUint16(localOffset + 26, true);
    entries.push({ name, crc, data: decoder.decode(new Uint8Array(buffer, dataStart, size)) });
    pos += 46 + nameLength;
  }
  return entries;
}

describe("zip", () => {
  it("computes the standard CRC-32", () => {
    expect(crc32(bytes("123456789"))).toBe(0xcbf43926);
  });

  it("writes entries an unzip tool can read back", async () => {
    const zip = createZip([
      { name: "a.png", data: bytes("first") },
      { name: "画像/b.png", data: bytes("second image") },
    ]);
    expect(zip.type).toBe("application/zip");
    const entries = readZip(await zip.arrayBuffer());
    expect(entries.map(({ name, data }) => [name, data])).toEqual([
      ["a.png", "first"],
      ["画像/b.png", "second image"],
    ]);
    expect(entries[0]!.crc).toBe(crc32(bytes("first")));
  });

  it("makes file names unique", () => {
    expect(uniqueNames(["a.png", "b.png", "a.png", "a.png", "c"])).toEqual(["a.png", "b.png", "a-2.png", "a-3.png", "c"]);
  });
});
