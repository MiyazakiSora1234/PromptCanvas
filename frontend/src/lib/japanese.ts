// Same character ranges the server uses to decide what to translate (hiragana, katakana, kanji, half-width kana).
const JAPANESE = /[぀-ヿ㐀-鿿ｦ-ﾟ]/;

export function hasJapanese(text: string): boolean {
  return JAPANESE.test(text);
}
