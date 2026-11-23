/**
 * Utility Helper Functions
 * Common utility functions used throughout the app
 */

import { MODELS, MODEL_LABELS } from './config';

/**
 * Map frontend model name to backend model name
 * @param {string} frontendModel - Frontend model identifier
 * @returns {string} Backend model identifier
 */
export const getBackendModelName = (frontendModel) => {
  const modelMap = {
    [MODELS.LLAMA]: MODELS.LLAMA,
    [MODELS.DEEPSEEK]: MODELS.DEEPSEEK,
    [MODELS.GPT_OSS]: MODELS.GPT_OSS
  };
  return modelMap[frontendModel] || frontendModel;
};

/**
 * Get display label for model
 * @param {string} modelId - Model identifier
 * @returns {string} Display label
 */
export const getModelLabel = (modelId) => {
  return MODEL_LABELS[modelId] || modelId;
};

/**
 * Validate file for upload
 * @param {File} file - File to validate
 * @param {Object} config - File configuration
 * @returns {Object} Validation result { isValid: boolean, error?: string }
 */
export const validateFile = (file, config) => {
  if (!file) {
    return { isValid: false, error: 'No file selected' };
  }

  if (file.size > config.MAX_SIZE) {
    return { isValid: false, error: 'File size must be less than 10MB' };
  }

  const nameLower = (file.name || '').toLowerCase();
  const ext = nameLower.includes('.') 
    ? nameLower.slice(nameLower.lastIndexOf('.')) 
    : '';

  const isTypeAllowed = config.ALLOWED_TYPES.includes(file.type);
  const isExtAllowed = config.ALLOWED_EXTENSIONS.includes(ext);

  if (!isTypeAllowed && !isExtAllowed) {
    return { 
      isValid: false, 
      error: 'Supported file types: PDF, TXT, DOC, DOCX, XLSX, XLS, CSV' 
    };
  }

  return { isValid: true };
};

/**
 * Format date to readable string
 * @param {string|Date} date - Date to format
 * @returns {string} Formatted date
 */
export const formatDate = (date) => {
  try {
    return new Date(date).toLocaleDateString();
  } catch {
    return 'Unknown date';
  }
};

/**
 * Parse streaming response chunks
 * @param {string} data - JSON string data
 * @returns {Object} Parsed data or null
 */
export const parseStreamChunk = (data) => {
  try {
    if (data === '[DONE]') {
      return { isDone: true };
    }
    return JSON.parse(data);
  } catch {
    return null;
  }
};

/**
 * Extract content from parsed chunk
 * @param {Object} parsed - Parsed chunk data
 * @returns {Object} { content: string, reasoning: string }
 */
export const extractChunkContent = (parsed) => {
  if (!parsed) return { content: '', reasoning: '' };

  let content = '';
  let reasoning = '';

  // Legacy wrapper: { token: "..." }
  if (parsed.token) {
    content = parsed.token;
  }
  // Raw model gateway chunk: choices[0].delta
  else if (parsed.choices?.length > 0) {
    const delta = parsed.choices[0].delta || {};
    if (delta.reasoning_content) {
      reasoning = delta.reasoning_content;
    }
    if (delta.content) {
      content = delta.content;
    }
    if (parsed.reasoning_content) {
      reasoning = parsed.reasoning_content;
    }
    if (parsed.content) {
      content = parsed.content;
    }
  }
  // Top-level reasoning_content field
  else if (parsed.reasoning_content) {
    reasoning = parsed.reasoning_content;
  }

  return { content, reasoning };
};

/**
 * Scroll element to bottom
 * @param {React.MutableRefObject} ref - Ref to element
 */
export const scrollToElement = (ref) => {
  if (ref?.current) {
    ref.current.scrollIntoView({ behavior: 'smooth' });
  }
};

/**
 * Generate unique ID
 * @returns {number} Unique ID
 */
export const generateId = () => {
  return Date.now() + Math.random();
};

/**
 * Check if device is mobile
 * @returns {boolean} Is mobile device
 */
export const isMobileDevice = () => {
  return window.innerWidth <= 768;
};
