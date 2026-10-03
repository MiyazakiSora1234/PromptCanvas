/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** API origin when the frontend is hosted separately, e.g. "http://localhost:8000". */
  readonly VITE_API_BASE?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
