/**
 * API Utilities
 * Handles all API calls with proper error handling
 */

import { API_BASE_URL, AUTH_API_URL } from './config';

/**
 * Create initial thread for new chat session
 * @param {string} userEmail - User email
 * @returns {Promise<string>} Thread ID
 */
export const createInitialThread = async (userEmail) => {
  if (!userEmail || userEmail === 'undefined' || userEmail.trim() === '') {
    throw new Error('Valid email is required to create thread');
  }

  try {
    const response = await fetch(`${API_BASE_URL}/chat/create_thread`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        email: userEmail.trim(),
        title: 'New Chat'
      })
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();
    if (!data.thread_id) {
      throw new Error('No thread ID received from server');
    }

    return data.thread_id;
  } catch (error) {
    console.error('Error creating thread:', error);
    throw error;
  }
};

/**
 * Load all threads for a user
 * @param {string} userEmail - User email
 * @returns {Promise<Array>} Array of threads
 */
export const loadUserThreads = async (userEmail) => {
  if (!userEmail || userEmail === 'undefined' || userEmail.trim() === '') {
    return [];
  }

  try {
    const response = await fetch(
      `${API_BASE_URL}/chat/threads?email=${encodeURIComponent(userEmail.trim())}`
    );
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();
    return Array.isArray(data) ? data : [];
  } catch (error) {
    console.error('Error loading threads:', error);
    return [];
  }
};

/**
 * Load chat history for a thread
 * @param {string} threadId - Thread ID
 * @returns {Promise<Array>} Array of messages
 */
export const loadThreadHistory = async (threadId) => {
  if (!threadId) return [];

  try {
    const response = await fetch(
      `${API_BASE_URL}/chat/history?thread_id=${threadId}`
    );
    
    if (!response.ok) {
      throw new Error(`Failed to load history: ${response.status}`);
    }
    
    const data = await response.json();
    
    if (!Array.isArray(data)) return [];
    
    return data.map((msg, index) => ({
      role: msg.role,
      content: msg.content,
      timestamp: msg.created_at,
      id: msg.id || `${msg.created_at}-${index}`
    }));
  } catch (error) {
    console.error('Error loading history:', error);
    return [];
  }
};

/**
 * Fetch current user information
 * @returns {Promise<Object>} User data
 */
export const fetchUserInfo = async () => {
  try {
    const response = await fetch(`${AUTH_API_URL}/api/user/me`, {
      credentials: 'include'
    });
    
    if (!response.ok) {
      if (response.status === 401) {
        return null;
      }
      throw new Error(`Auth check failed: ${response.status}`);
    }

    const data = await response.json();
    
    if (data && data.email && data.email !== 'undefined' && data.email.trim() !== '') {
      return data;
    }
    
    throw new Error('Invalid user data received from server');
  } catch (error) {
    console.error('Error fetching user info:', error);
    throw error;
  }
};

/**
 * Handle email login/register
 * @param {string} email - User email
 * @param {string} password - User password
 * @param {boolean} isRegister - Whether this is registration
 * @returns {Promise<Object>} Response data
 */
export const handleEmailAuth = async (email, password, isRegister) => {
  const endpoint = isRegister 
    ? `${AUTH_API_URL}/api/register` 
    : `${AUTH_API_URL}/api/login`;
  
  try {
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include',
      body: JSON.stringify({ 
        email, 
        password, 
        name: email.split('@')[0] 
      })
    });
    
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.message || data.detail || 'Authentication failed');
    }
    
    return data;
  } catch (error) {
    console.error('Auth error:', error);
    throw error;
  }
};

/**
 * Upload document to thread
 * @param {string} threadId - Thread ID
 * @param {File} file - File to upload
 * @returns {Promise<Object>} Upload response
 */
export const uploadDocument = async (threadId, file) => {
  const formData = new FormData();
  formData.append('thread_id', threadId);
  formData.append('file', file);

  try {
    const response = await fetch(`${API_BASE_URL}/chat/upload_doc`, {
      method: 'POST',
      body: formData
    });
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Upload failed: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error uploading file:', error);
    throw error;
  }
};

/**
 * Update thread title
 * @param {string} threadId - Thread ID
 * @param {string} newTitle - New title for the thread
 * @returns {Promise<Object>} Response from server
 */
export const updateThreadTitle = async (threadId, newTitle) => {
  if (!threadId || !newTitle || newTitle.trim() === '') {
    throw new Error('Thread ID and title are required');
  }

  try {
    const response = await fetch(`${API_BASE_URL}/chat/update_thread_title`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        thread_id: threadId,
        title: newTitle.trim()
      })
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      console.warn('Failed to update thread title:', errorData);
      // Don't throw - this is a non-critical operation
      return null;
    }

    return await response.json();
  } catch (error) {
    console.warn('Error updating thread title:', error);
    // Don't throw - this is a non-critical operation
    return null;
  }
};

/**
 * Send message to chat endpoint
 * Routes to RAG endpoint if document is provided, otherwise uses regular chat endpoint
 * @param {string} threadId - Thread ID
 * @param {string} message - Message content
 * @param {string} model - Model to use
 * @param {Object} documentInfo - Document info with {id, filename, chunks} (optional)
 * @param {AbortSignal} abortSignal - AbortSignal to cancel request (optional)
 * @returns {Promise<ReadableStream>} Streaming response
 */
export const sendChatMessage = async (threadId, message, model, documentInfo = null, abortSignal = null) => {
  let endpoint, payload;
  
  // Determine which endpoint to use based on document availability
  if (documentInfo && documentInfo.id) {
    // Document is loaded - use RAG endpoint
    endpoint = `${API_BASE_URL}/chat/rag`;
    payload = {
      thread_id: threadId,
      document_id: documentInfo.id,
      query: message,
      model: model
    };
    console.log('[RAG] Using RAG endpoint with document:', documentInfo.id);
  } else {
    // No document - use regular chat endpoint
    endpoint = `${API_BASE_URL}/chat/send`;
    payload = {
      thread_id: threadId,
      message: message,
      model: model
    };
    console.log('[CHAT] Using regular chat endpoint (no document)');
  }

  try {
    const fetchOptions = {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    };

    // Include abort signal if provided
    if (abortSignal) {
      fetchOptions.signal = abortSignal;
    }

    const response = await fetch(endpoint, fetchOptions);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Request failed: ${response.status}`);
    }

    if (!response.body) {
      throw new Error('No response body received');
    }

    return response.body;
  } catch (error) {
    console.error('[API Error]', endpoint, error);
    throw error;
  }
};
