// ============================================================
// config.js  —  Single backend (auth + chat merged into server.py)
// ============================================================

// Both auth and chat API now live on the same FastAPI server.
// AUTH_API_URL is kept as an alias so existing component imports
// (e.g. UserCircle's logout link) continue to work unchanged.
export const API_BASE_URL  = process.env.REACT_APP_API_BASE_URL  || "http://dev.nmit.ac.in:9096";
export const AUTH_API_URL  = process.env.REACT_APP_AUTH_API_URL  || API_BASE_URL;   // same server

// ---- Model keys ----
export const MODELS = {
  LLAMA:    "llama",
  DEEPSEEK: "deepseek",
  GPT_OSS:  "gpt-oss",
};

export const MODEL_LABELS = {
  [MODELS.LLAMA]:    "Llama 3.1 8B",
  [MODELS.DEEPSEEK]: "DeepSeek Coder V2",
  [MODELS.GPT_OSS]:  "GPT-OSS 20B",
};

// ---- File upload constraints ----
export const FILE_CONFIG = {
  MAX_SIZE_MB:       50,
  ALLOWED_TYPES:     [".pdf", ".docx", ".txt", ".md"],
  ALLOWED_MIME_TYPES: [
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
    "text/markdown",
  ],
};

// ---- UI defaults ----
export const UI_CONFIG = {
  MAX_THREADS_DISPLAYED: 50,
  MESSAGE_HISTORY_LIMIT: 100,
  TOAST_DURATION_MS:     3000,
  SCROLL_DEBOUNCE_MS:    100,
};