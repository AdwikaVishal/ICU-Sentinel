/// <reference types="vitest/config" />

declare interface ImportMetaEnv {
  readonly VITE_P3_API_BASE?: string;
}

declare interface ImportMeta {
  readonly env: ImportMetaEnv;
}
