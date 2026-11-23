

// API Configuration
// export const AUTH_API_URL ="http://36.255.14.28:9095";
// export const API_BASE_URL = 'http://36.255.14.28:9096';
// export const AUTH_API_URL = 'http://192.168.190.28:9095';
// export const API_BASE_URL = 'http://192.168.190.28:9096';
export const AUTH_API_URL = process.env.REACT_APP_AUTH_API_URL || 'http://localhost:9096';
export const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:9096';

// Model Configuration
export const MODELS = {
  LLAMA: 'llama',
  DEEPSEEK: 'deepseek',
  GPT_OSS: 'gpt-oss'
};

export const MODEL_LABELS = {
  [MODELS.LLAMA]: 'Llama 3.1 8B',
  [MODELS.DEEPSEEK]: 'DeepSeek Coder V2',
  [MODELS.GPT_OSS]: 'GPT-OSS'
};

// File Upload Configuration
export const FILE_CONFIG = {
  MAX_SIZE: 10 * 1024 * 1024, // 10MB
  ALLOWED_TYPES: [
    'application/pdf',
    'text/plain',
    'application/msword',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'application/vnd.ms-excel',
    'text/csv'
  ],
  ALLOWED_EXTENSIONS: ['.pdf', '.txt', '.doc', '.docx', '.xlsx', '.xls', '.csv']
};

// UI Configuration
export const UI_CONFIG = {
  TOAST_DURATION: 5000,
  SCROLL_DEBOUNCE: 100,
  BUFFER_FLUSH_INTERVAL: 50,
  MESSAGES_VIRTUALIZATION_THRESHOLD: 50
};
