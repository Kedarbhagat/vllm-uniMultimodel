import React, { useState, useEffect, useCallback, useRef } from 'react';
import { MessageCircle, AlertCircle } from 'lucide-react';

// ============================================
// COMPONENT IMPORTS
// ============================================
import ErrorBoundary from './components/ErrorBoundary';
import Toast from './components/Toast';
import SidebarHeader from './components/SidebarHeader';
import NewChatButton from './components/NewChatButton';
import DarkModeToggle from './components/DarkModeToggle';
import UserCircle from './components/UserCircle';
import UserProfile from './components/UserProfile';
import MessageList from './components/MessageList';
import ChatInput from './components/ChatInput';
import LoginScreen from './components/LoginScreen';

// ============================================
// HOOK IMPORTS
// ============================================
import {
  useToast,
  useScrollToBottom,
  useDarkMode,
  useOutsideClick
} from './hooks';

// ============================================
// API IMPORTS
// ============================================
import {
  fetchUserInfo,
  createInitialThread,
  loadUserThreads,
  loadThreadHistory,
  uploadDocument,
  sendChatMessage,
  updateThreadTitle
} from './api';

// ============================================
// UTILITY IMPORTS
// ============================================
import {
  getBackendModelName,
  getModelLabel,
  validateFile,
  formatDate,
  parseStreamChunk,
  extractChunkContent,
  generateId
} from './utils';

// ============================================
// CONFIG IMPORTS
// ============================================
import {
  AUTH_API_URL,
  MODELS,
  MODEL_LABELS,
  FILE_CONFIG,
  UI_CONFIG
} from './config';

import './App.css';

/**
 * Main ChatFrontend Component
 * 
 * This is the primary application component that:
 * - Manages authentication and user sessions
 * - Orchestrates all child components
 * - Handles state management for chat messages and threads
 * - Manages API interactions for sending/receiving messages
 * - Handles file uploads and document processing
 * 
 * Key responsibilities:
 * 1. Authentication check on mount
 * 2. User session initialization
 * 3. Message streaming and display
 * 4. Thread management
 * 5. Model selection
 * 6. Dark mode management
 * 7. File upload handling
 * 8. Error handling and display
 * 
 * @returns {JSX.Element} The main chat application interface
 */
const ChatFrontend = () => {
  // ============================================
  // STATE MANAGEMENT
  // ============================================

  // Chat and message state
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [threads, setThreads] = useState([]);
  const [activeThread, setActiveThread] = useState(null);
  const [abortController, setAbortController] = useState(null);

  // User authentication state
  const [userEmail, setUserEmail] = useState('');
  const [userInfo, setUserInfo] = useState(null);
  const [isAuthLoading, setIsAuthLoading] = useState(true);
  const [authError, setAuthError] = useState(null);

  // UI state
  const [selectedModel, setSelectedModel] = useState(MODELS.GPT_OSS);
  const [threadDocuments, setThreadDocuments] = useState(new Map());
  const [currentDocumentInfo, setCurrentDocumentInfo] = useState(null); // Track current document directly
  const [showSettings, setShowSettings] = useState(false);
  const [initializationError, setInitializationError] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [showProfile, setShowProfile] = useState(false);
  const [threadTitleUpdated, setThreadTitleUpdated] = useState(new Map()); // Track which threads have been auto-renamed

  // ============================================
  // CUSTOM HOOKS
  // ============================================

  // Toast notifications
  const { toast, showToast, closeToast } = useToast();

  // Dark mode management
  const { darkMode, setDarkMode } = useDarkMode();

  // Scroll to bottom on new messages
  const { messagesEndRef } = useScrollToBottom(messages.length);

  // ============================================
  // REF MANAGEMENT
  // ============================================

  // Sidebar reference for outside click detection
  const sidebarRef = useRef(null);
  const toggleButtonRef = useRef(null);

  // ============================================
  // MEMOIZED VALUES
  // ============================================

  /**
   * Get the document information for the currently active thread
   * Uses direct state variable for immediate access
   */
  const getCurrentThreadDocument = currentDocumentInfo;

  /**
   * Generate a meaningful title from conversation messages
   * Takes the first user message and creates a short, descriptive title
   * @param {Array} messageList - Array of messages
   * @returns {string} Generated title
   */
  const generateTitleFromConversation = (messageList) => {
    // Find the first user message
    const firstUserMessage = messageList.find(msg => msg.role === 'human' || msg.role === 'user');
    
    if (!firstUserMessage || !firstUserMessage.content) {
      return 'New Chat';
    }

    // Take first 50 characters and clean up
    let title = firstUserMessage.content
      .trim()
      .substring(0, 50)
      .replace(/\n/g, ' ')
      .replace(/\s+/g, ' ');

    // Add ellipsis if truncated
    if (firstUserMessage.content.length > 50) {
      title += '...';
    }

    // Ensure we have a valid title
    return title || 'New Chat';
  };

  // ============================================
  // EFFECTS
  // ============================================

  /**
   * EFFECT 1: Check authentication on component mount
   * 
   * Purpose: Verify if user is already authenticated
   * Cleanup: Set cleanup flag to prevent state updates on unmount
   */
  useEffect(() => {
    let isMounted = true;

    const checkAuth = async () => {
      setIsAuthLoading(true);
      setAuthError(null);

      try {
        // Attempt to fetch current user info
        // This will throw if user is not authenticated
        const userData = await fetchUserInfo();

        if (isMounted && userData) {
          setUserInfo(userData);
          setUserEmail(userData.email.trim());
        }
      } catch (error) {
        if (isMounted) {
          console.error('Auth check failed:', error);
          setAuthError(error.message);
          // Don't set email - user will need to login
        }
      } finally {
        if (isMounted) {
          setIsAuthLoading(false);
        }
      }
    };

    checkAuth();

    // Cleanup function
    return () => {
      isMounted = false;
    };
  }, []);

  /**
   * EFFECT 2: Initialize user session when email changes
   * 
   * Purpose: Load user's threads and set up the chat environment
   * Dependencies: Only runs when userEmail changes
   * Cleanup: Set cleanup flag to prevent state updates on unmount
   */
  useEffect(() => {
    let isMounted = true;

    const initializeSession = async () => {
      // Don't proceed if no email
      if (!userEmail) return;


      setInitializationError(null);

      try {

        // Load all threads for this user
        const userThreads = await loadUserThreads(userEmail);

        if (!isMounted) return;

        setThreads(userThreads);

        if (userThreads && userThreads.length > 0) {
          // Sort by most recent first
          const sorted = userThreads.sort((a, b) =>
            new Date(b.created_at || b.timestamp || 0) -
            new Date(a.created_at || a.timestamp || 0)
          );

          // Load the most recent thread
          const mostRecent = sorted[0];
          setActiveThread(mostRecent.thread_id);

          // Mark all existing threads as already having titles
          setThreadTitleUpdated(prev => {
            const newMap = new Map(prev);
            userThreads.forEach(thread => {
              newMap.set(thread.thread_id, true);
            });
            return newMap;
          });

          // Load the chat history for this thread
          const history = await loadThreadHistory(mostRecent.thread_id);
          if (isMounted) {
            setMessages(history || []);
          }

          // Add welcome message
          if (isMounted) {
            setMessages(prev => [...prev, {
              role: 'system',
              content: `Welcome back! Loaded your chat: "${mostRecent.title}"`,
              timestamp: new Date().toISOString(),
              id: generateId()
            }]);
          }
        } else {
          // Create initial thread for new user
          const threadId = await createInitialThread(userEmail);
          if (isMounted) {
            setActiveThread(threadId);
            setMessages([]);

            // Add welcome message
            setMessages(prev => [...prev, {
              role: 'system',
              content: 'Welcome! Created your first chat session!',
              timestamp: new Date().toISOString(),
              id: generateId()
            }]);
          }
        }

        // Clear any previous documents
        if (isMounted) {
          setThreadDocuments(new Map());
        }
      } catch (error) {
        if (isMounted) {
          console.error('Session init failed:', error);
          setInitializationError(error.message);
          showToast(`Failed to initialize: ${error.message}`);
        }
      }
    };

    initializeSession();

    // Cleanup function
    return () => {
      isMounted = false;
    };
  }, [userEmail, showToast]);

  /**
   * Stop the current LLM response stream
   * 
   * Purpose: Allow user to abort/cancel streaming response
   * Aborts the fetch request and cleans up state
   */
  const handleStopResponse = useCallback(() => {
    if (abortController) {
      abortController.abort();
      setAbortController(null);
      setIsLoading(false);
      showToast('Response stopped', 'info');
    }
  }, [abortController, showToast]);

  /**
   * EFFECT 3: Handle sidebar close on outside click
   * 
   * Purpose: Close sidebar when user clicks outside of it
   * Dependencies: sidebarOpen state
   */
  useOutsideClick(sidebarRef, () => setSidebarOpen(false), sidebarOpen);

  /**
   * EFFECT 4: Handle keyboard shortcuts (Escape to stop response)
   * 
   * Purpose: Allow user to stop response with Escape key
   * Dependencies: isLoading state
   */
  useEffect(() => {
    const handleKeyPress = (e) => {
      if (e.key === 'Escape' && isLoading) {
        handleStopResponse();
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [isLoading, handleStopResponse]);

  // ============================================
  // CALLBACK FUNCTIONS
  // ============================================

  /**
   * Check and update thread title if needed
   * Triggers after first user message or after 5+ messages
   * 
   * @param {string} threadId - Thread ID to update
   * @param {Array} messageList - Current messages in thread
   */
  const checkAndUpdateThreadTitle = useCallback(async (threadId, messageList) => {
    // Don't update if already updated for this thread
    if (threadTitleUpdated.get(threadId)) {
      return;
    }

    // Don't update if we don't have at least one user message
    const userMessages = messageList.filter(msg => msg.role === 'human' || msg.role === 'user');
    if (userMessages.length === 0) {
      return;
    }

    // Generate new title from conversation
    const newTitle = generateTitleFromConversation(messageList);

    // Skip if title is still generic
    if (newTitle === 'New Chat') {
      return;
    }

    try {
      // Update on backend
      await updateThreadTitle(threadId, newTitle);

      // Update local thread list
      setThreads(prev =>
        prev.map(thread =>
          thread.thread_id === threadId
            ? { ...thread, title: newTitle }
            : thread
        )
      );

      // Mark this thread as updated
      setThreadTitleUpdated(prev => new Map(prev).set(threadId, true));

      console.log('[TITLE_UPDATE] ✓ Thread title updated:', newTitle);
    } catch (error) {
      console.warn('[TITLE_UPDATE] Failed to update title:', error);
      // Not critical - continue anyway
    }
  }, [threadTitleUpdated, generateTitleFromConversation]);

  /**
   * Set the document for the currently active thread
   * 
   * @param {Object} documentInfo - Document metadata {id, filename, chunks}
   * @param {string} documentInfo.id - Document ID from backend
   * @param {string} documentInfo.filename - Document file name
   * @param {number} documentInfo.chunks - Number of chunks processed
   */
  const setCurrentThreadDocument = useCallback((documentInfo) => {
    if (!activeThread) {
      console.log('[SET_DOC] ✗ activeThread is null, cannot set document');
      return;
    }
    console.log('[SET_DOC] Setting document for thread:', activeThread, documentInfo);
    
    // Update direct state immediately for real-time access
    setCurrentDocumentInfo(documentInfo);
    
    // Also keep the Map for historical tracking
    setThreadDocuments(prev => {
      const newMap = new Map(prev);
      if (documentInfo) {
        newMap.set(activeThread, documentInfo);
        console.log('[SET_DOC] ✓ Document added to Map. New size:', newMap.size);
      } else {
        newMap.delete(activeThread);
        console.log('[SET_DOC] ✓ Document removed from Map. New size:', newMap.size);
      }
      return newMap;
    });
  }, [activeThread]);

  /**
   * Create a new chat thread and switch to it
   * 
   * Purpose: Initialize a new conversation
   * Dependencies: userEmail, showToast function
   */
  const handleCreateNewThread = useCallback(async () => {
    if (!userEmail) {
      showToast('Please log in to create a new chat');
      return;
    }

    try {
      const threadId = await createInitialThread(userEmail);
      if (!threadId) throw new Error('No thread ID returned');

      setActiveThread(threadId);
      setThreadDocuments(new Map()); // Clear documents for new thread
      setCurrentDocumentInfo(null); // Clear current document
      
      // Reset title update flag for new thread
      setThreadTitleUpdated(prev => {
        const newMap = new Map(prev);
        newMap.delete(threadId);
        return newMap;
      });

      // Set initial message in one operation
      setMessages([{
        role: 'system',
        content: 'New chat session started!',
        timestamp: new Date().toISOString(),
        id: generateId()
      }]);

      const updatedThreads = await loadUserThreads(userEmail);
      if (updatedThreads) {
        setThreads(updatedThreads);
      }
      setSidebarOpen(false);
    } catch (error) {
      console.error('Create thread error:', error);
      showToast(`Failed to create chat: ${error.message}`);
    }
  }, [userEmail, showToast]);

  /**
   * Switch to a different thread
   * 
   * @param {string} threadId - The thread ID to switch to
   * Dependencies: activeThread, showToast function
   */
  const handleThreadSwitch = useCallback(async (threadId) => {
    if (!threadId) return; // Defensive check
    if (threadId === activeThread) return;
    setActiveThread(threadId);
    
    // Restore document from Map if it exists for this thread, otherwise clear it
    const docForThread = threadDocuments.get(threadId) || null;
    setCurrentDocumentInfo(docForThread);
    console.log('[THREAD_SWITCH] Switched to thread:', threadId, 'Document:', docForThread);

    try {
      const history = await loadThreadHistory(threadId);
      setMessages(history || []);
      setSidebarOpen(false);

      // Check if we need to auto-rename this thread
      if (history && history.length > 0) {
        checkAndUpdateThreadTitle(threadId, history);
      }
    } catch (error) {
      console.error('Load history error:', error);
      showToast(`Failed to load chat: ${error.message}`);
    }
  }, [activeThread, showToast, threadDocuments, checkAndUpdateThreadTitle]);

  /**
   * Handle file upload for the active thread
   * 
   * @param {File} file - The file to upload
   * 
   * Process:
   * 1. Validate file size and type
   * 2. Send to backend for processing
   * 3. Display document info in sidebar
   * 4. Show success/error messages
   */
  const handleFileUpload = useCallback(async (file) => {
    if (!activeThread) {
      showToast('Please start a conversation first');
      return;
    }

    // Validate file before uploading
    const validation = validateFile(file, FILE_CONFIG);
    if (!validation.isValid) {
      showToast(validation.error);
      return;
    }

    try {
      setIsUploading(true);

      // Show uploading message
      setMessages(prev => [...prev, {
        role: 'system',
        content: `Uploading "${file.name}"...`,
        timestamp: new Date().toISOString(),
        id: generateId()
      }]);

      // Upload the document
      console.log('[UPLOAD] Starting upload for thread:', activeThread);
      const result = await uploadDocument(activeThread, file);
      console.log('[UPLOAD] Server response:', result);

      if (result && result.document_id) {
        // Store document info
        console.log('[UPLOAD] ✓ Document has ID, calling setCurrentThreadDocument');
        setCurrentThreadDocument({
          id: result.document_id,
          filename: result.filename,
          chunks: result.chunks
        });
      } else {
        console.log('[UPLOAD] ✗ ERROR: No document_id in response!', result);
      }
      
      if (result && result.document_id) {

        // Show success message
        console.log('[UPLOAD] Setting document for activeThread:', activeThread, {
          id: result.document_id,
          filename: result.filename,
          chunks: result.chunks
        });
        
        setMessages(prev => [...prev, {
          role: 'system',
          content: `Document "${result.filename}" uploaded successfully. ${result.chunks} chunks processed.`,
          timestamp: new Date().toISOString(),
          id: generateId()
        }]);

        showToast('Document uploaded successfully!', 'success');
      }
    } catch (error) {
      console.error('Upload error:', error);

      // Show error message
      setMessages(prev => [...prev, {
        role: 'system',
        content: `Failed to upload: ${error.message}`,
        timestamp: new Date().toISOString(),
        id: generateId()
      }]);

      showToast(`Upload failed: ${error.message}`);
    } finally {
      setIsUploading(false);
    }
  }, [activeThread, showToast, setCurrentThreadDocument]);



  /**
   * Send a message and handle streaming response
   * 
   * @param {string} messageContent - The user's message
   * 
   * Process:
   * 1. Add user message to chat
   * 2. Start streaming response from backend
   * 3. Parse and display streaming chunks
   * 4. Handle reasoning sections
   * 5. Complete message and update UI
   */
  const handleSendMessage = useCallback(async (messageContent) => {
    // Validate inputs
    if (!messageContent || !activeThread || isLoading) return;

    // Create and display user message
    const userMessage = {
      role: 'human',
      content: messageContent,
      timestamp: new Date().toISOString(),
      id: generateId()
    };

    // Create new abort controller for this request
    const controller = new AbortController();
    setAbortController(controller);

    setIsLoading(true);
    setMessages(prev => [...prev, userMessage]);

    try {
      // Get current document info for this thread
      const documentInfo = getCurrentThreadDocument;
      
      // Log which mode we're using
      console.log('[SEND] activeThread:', activeThread);
      console.log('[SEND] threadDocuments Map size:', threadDocuments.size);
      console.log('[SEND] threadDocuments contents:', Array.from(threadDocuments.entries()));
      console.log('[SEND] getCurrentThreadDocument:', documentInfo);
      
      if (documentInfo && documentInfo.id) {
        console.log('[SEND] ✓ RAG mode - Using document:', documentInfo.filename);
      } else {
        console.log('[SEND] ✗ Chat mode - No document loaded');
      }

      // Send message and get streaming response
      const responseBody = await sendChatMessage(
        activeThread,
        messageContent,
        getBackendModelName(selectedModel),
        documentInfo,  // Explicitly pass document info
        controller.signal  // Pass abort signal to fetch
      );

      // Set up streaming reader
      const reader = responseBody.getReader();
      const decoder = new TextDecoder();
      let assistantContent = '';
      let assistantReasoning = '';

      // Create placeholder for assistant message
      const assistantMessageId = generateId();
      const assistantMessage = {
        role: 'ai',
        content: '',
        reasoning: '',
        timestamp: new Date().toISOString(),
        streaming: true,
        id: assistantMessageId
      };

      setMessages(prev => [...prev, assistantMessage]);

      // Buffer for batching updates
      let contentBuffer = '';
      let reasoningBuffer = '';
      let lastUpdate = Date.now();

      /**
       * Flush buffered content to message
       * Called periodically to update UI with new content
       */
      const flushBuffers = () => {
        if (!contentBuffer && !reasoningBuffer) return;

        // Update the message with buffered content
        setMessages(prev =>
          prev.map(msg =>
            msg.id === assistantMessageId
              ? {
                ...msg,
                content: assistantContent + contentBuffer,
                reasoning: assistantReasoning + reasoningBuffer
              }
              : msg
          )
        );

        // Move buffers to accumulated content
        assistantContent += contentBuffer;
        assistantReasoning += reasoningBuffer;
        contentBuffer = '';
        reasoningBuffer = '';
        lastUpdate = Date.now();
      };

      // Read streaming response chunks
      let shouldContinueReading = true;
      while (shouldContinueReading) {
        let chunkValue = null;
        try {
          const { done, value } = await reader.read();
          if (done) break;
          chunkValue = value;
        } catch (error) {
          // Handle abort error gracefully
          if (error.name === 'AbortError') {
            // Flush any remaining content
            flushBuffers();
            // Mark message as stopped
            setMessages(prev =>
              prev.map(msg =>
                msg.id === assistantMessageId
                  ? {
                    ...msg,
                    streaming: false,
                    content: assistantContent + contentBuffer + '\n\n[Response stopped by user]'
                  }
                  : msg
              )
            );
            setIsLoading(false);
            setAbortController(null);
            return;
          }
          throw error;
        }

        if (!chunkValue) continue;

        // Decode the chunk
        const chunk = decoder.decode(chunkValue, { stream: true });
        const lines = chunk.split('\n');

        // Process each line
        for (const line of lines) {
          const trimmed = line.trim();

          // Skip empty lines and non-data lines
          if (!trimmed || !trimmed.startsWith('data:')) continue;

          // Extract and parse the data
          const data = trimmed.slice(5).trim();
          const parsed = parseStreamChunk(data);

          // Skip completion markers and invalid data
          if (parsed?.isDone) continue;
          if (!parsed) continue;

          // Extract content and reasoning from chunk
          const { content, reasoning } = extractChunkContent(parsed);
          contentBuffer += content;
          reasoningBuffer += reasoning;

          // Flush buffers periodically
          const now = Date.now();
          if (reasoningBuffer.length > 0 || now - lastUpdate > UI_CONFIG.BUFFER_FLUSH_INTERVAL) {
            flushBuffers();
          }
        }
      }

      // Final flush to ensure all content is displayed
      flushBuffers();

      // Mark message as complete (not streaming)
      setMessages(prev =>
        prev.map(msg =>
          msg.id === assistantMessageId
            ? {
              ...msg,
              content: assistantContent,
              reasoning: assistantReasoning,
              streaming: false
            }
            : msg
        )
      );

      // Check if we should auto-rename the thread after receiving response
      setMessages(prev => {
        // Call title check with the updated messages
        setTimeout(() => {
          checkAndUpdateThreadTitle(activeThread, prev);
        }, 0);
        return prev;
      });
    } catch (error) {
      console.error('Send message error:', error);

      // Don't show error for abort (eslint-disable-next-line no-unsafe-optional-chaining)
      if (error?.name !== 'AbortError') {
        // Add error message to chat
        setMessages(prev => [...prev, {
          role: 'system',
          content: `Error: ${error.message}`,
          timestamp: new Date().toISOString(),
          id: generateId()
        }]);

        showToast(`Failed to send message: ${error.message}`);
      }
    } finally {
      setIsLoading(false);
      setAbortController(null);
    }
  }, [activeThread, isLoading, selectedModel, showToast, getCurrentThreadDocument, threadDocuments, checkAndUpdateThreadTitle]);

  // ============================================
  // RENDER: LOADING STATE
  // ============================================

  if (isAuthLoading) {
    return (
      <div className="flex flex-col items-center justify-center h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-zinc-900 dark:to-zinc-950">
        <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mb-4"></div>
        <p className="text-gray-600 dark:text-zinc-400">Checking authentication...</p>
      </div>
    );
  }

  // ============================================
  // RENDER: LOGIN SCREEN
  // ============================================

  if (!userEmail) {
    return (
      <LoginScreen
        onAuthSuccess={(email) => setUserEmail(email)}
        authError={authError}
        showErrorMessage={showToast}
      />
    );
  }

  // ============================================
  // RENDER: ERROR SCREEN
  // ============================================

  if (initializationError) {
    return (
      <div className="flex flex-col items-center justify-center h-screen bg-gradient-to-br from-red-50 to-red-100 dark:from-zinc-900 dark:to-zinc-950">
        <div className="bg-white dark:bg-zinc-800 p-8 rounded-lg shadow-lg flex flex-col items-center max-w-md">
          <AlertCircle className="w-16 h-16 text-red-500 mb-4" />
          <h2 className="text-2xl font-bold mb-2 text-red-700 dark:text-red-400">Initialization Failed</h2>
          <p className="mb-6 text-gray-600 dark:text-gray-400 text-center">{initializationError}</p>
          <button
            onClick={() => window.location.reload()}
            className="bg-red-600 hover:bg-red-700 text-white px-6 py-3 rounded-lg font-medium transition-colors dark:bg-red-500 dark:hover:bg-red-600"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  // ============================================
  // RENDER: MAIN CHAT INTERFACE
  // ============================================

  return (
    <ErrorBoundary>
      <div className="flex h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-zinc-900 dark:to-zinc-950 relative">

        {/* TOAST NOTIFICATION */}
        {toast && (
          <Toast
            message={toast.message}
            type={toast.type}
            onClose={closeToast}
          />
        )}

        {/* USER PROFILE MODAL */}
        {showProfile && (
          <UserProfile
            userInfo={userInfo}
            userEmail={userEmail}
            onClose={() => setShowProfile(false)}
            onPasswordUpdate={() => {
              showToast('Password changed successfully!', 'success');
            }}
            darkMode={darkMode}
          />
        )}

        {/* SIDEBAR OVERLAY (Mobile) */}
        {sidebarOpen && (
          <div
            className="fixed inset-0 bg-black bg-opacity-50 z-30"
            onClick={() => setSidebarOpen(false)}
            aria-label="Sidebar overlay"
          />
        )}

        {/* SIDEBAR */}
        <div
          ref={sidebarRef}
          className={`fixed top-0 left-0 h-full w-80 bg-white dark:bg-zinc-900 shadow-2xl border-r border-zinc-200 dark:border-zinc-800 flex flex-col transform transition-transform duration-300 ease-in-out z-40 ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'
            }`}
        >
          {/* Sidebar Header */}
          <SidebarHeader
            darkMode={darkMode}
            onClose={() => setSidebarOpen(false)}
          />

          {/* New Chat Button */}
          <NewChatButton
            darkMode={darkMode}
            isLoading={isLoading}
            onClick={handleCreateNewThread}
          />

          {/* Document Status */}
          {getCurrentThreadDocument && (
            <div className="mx-5 mb-5">
              <div className="bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-700 p-3">
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center text-green-800 dark:text-green-200 min-w-0">
                    <span className="text-sm font-medium truncate">
                      {getCurrentThreadDocument.filename}
                    </span>
                  </div>
                  <button
                    onClick={() => setCurrentThreadDocument(null)}
                    className="text-green-600 dark:text-green-300 hover:text-green-800 dark:hover:text-green-100 transition-colors flex-shrink-0"
                    aria-label="Remove document"
                    title="Remove document"
                  >
                    ✕
                  </button>
                </div>
                <div className="text-xs text-green-600 dark:text-green-400 mt-1">
                  {getCurrentThreadDocument.chunks} chunks processed
                </div>
              </div>
            </div>
          )}

          {/* Settings Section */}
          <div className="px-5 pb-5">
            <button
              onClick={() => setShowSettings(!showSettings)}
              className="w-full bg-zinc-100 dark:bg-zinc-800 text-zinc-700 dark:text-zinc-200 py-2 px-4 rounded-xl font-semibold hover:bg-zinc-200 dark:hover:bg-zinc-700 transition-all"
              aria-label="Settings"
            >
              Settings
            </button>

            {/* Settings Dropdown */}
            {showSettings && (
              <div className="mt-3 p-3 bg-zinc-50 dark:bg-zinc-900 rounded-lg border border-zinc-200 dark:border-zinc-700 space-y-3">
                <div>
                  <label className="block text-xs font-semibold text-zinc-700 dark:text-zinc-300 mb-1">
                    Model
                  </label>
                  <select
                    value={selectedModel}
                    onChange={(e) => {
                      setSelectedModel(e.target.value);
                      showToast(
                        `Model switched to ${getModelLabel(e.target.value)}`,
                        'success'
                      );
                    }}
                    className="w-full p-2 border border-zinc-300 dark:border-zinc-700 rounded-md text-base bg-white dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100 focus:outline-none focus:border-blue-500 dark:focus:border-blue-400"
                    aria-label="Model selection"
                  >
                    {Object.entries(MODEL_LABELS).map(([key, label]) => (
                      <option key={key} value={key}>
                        {label}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            )}
          </div>

          {/* Thread List */}
          <div className="flex-1 overflow-y-auto px-5">
            <h3 className="text-base font-bold text-zinc-700 dark:text-zinc-200 mb-3">
              Recent Conversations
            </h3>
            <div className="space-y-2">
              {threads.length === 0 ? (
                <p className="text-sm text-zinc-500 dark:text-zinc-400">
                  No conversations yet. Create a new one to get started!
                </p>
              ) : (
                threads.map((thread) => (
                  <button
                    key={thread.thread_id}
                    onClick={() => handleThreadSwitch(thread.thread_id)}
                    className={`w-full text-left p-3 rounded-xl transition-all ${activeThread === thread.thread_id
                        ? 'bg-gray-100 dark:bg-zinc-800 border border-gray-300 dark:border-zinc-700'
                        : 'bg-white dark:bg-zinc-900 border border-zinc-100 dark:border-zinc-700'
                      } hover:bg-gray-100 dark:hover:bg-zinc-800`}
                    title={thread.title}
                  >
                    <div className="text-base font-semibold text-zinc-800 dark:text-zinc-100 truncate">
                      {thread.title}
                    </div>
                    <div className="text-xs text-zinc-500 dark:text-zinc-400 mt-1">
                      {formatDate(thread.created_at || thread.timestamp)}
                    </div>
                  </button>
                ))
              )}
            </div>
          </div>
        </div>

        {/* MAIN CHAT AREA */}
        <div className="flex-1 flex flex-col bg-gradient-to-br from-white to-zinc-50 dark:from-zinc-950 dark:to-zinc-900">

          {/* HEADER */}
          <div className="bg-white dark:bg-zinc-900 shadow-sm border-b border-zinc-200 dark:border-zinc-800 p-6">
            <div className="flex items-center justify-between">

              {/* Left side: Sidebar toggle and title */}
              <div className="flex items-center gap-4">
                <button
                  ref={toggleButtonRef}
                  onClick={() => setSidebarOpen(!sidebarOpen)}
                  className="p-2 hover:bg-zinc-100 dark:hover:bg-zinc-800 rounded-lg transition-colors"
                  aria-label="Toggle sidebar"
                  title="Toggle sidebar"
                >
                  <MessageCircle className="w-5 h-5 text-zinc-600 dark:text-zinc-400" />
                </button>
                <div>
                  <p className="text-base text-zinc-600 dark:text-zinc-400">
                    {getCurrentThreadDocument
                      ? `Chatting about: ${getCurrentThreadDocument.filename}`
                      : threads.find(t => t.thread_id === activeThread)?.title ||
                      'New Chat'
                    }
                  </p>
                </div>
              </div>

              {/* Right side: Model, dark mode, and user menu */}
              <div className="flex items-center gap-5">

                {/* Dark Mode Toggle */}
                <DarkModeToggle
                  darkMode={darkMode}
                  onToggle={() => setDarkMode(v => !v)}
                />

                {/* Model Display */}
                <div className="text-base text-zinc-500 dark:text-zinc-300 whitespace-nowrap hidden sm:block">
                  Model: {getModelLabel(selectedModel)}
                </div>

                {/* User Circle / Logout */}
                <UserCircle
                  userInfo={userInfo}
                  userEmail={userEmail}
                  authApiUrl={AUTH_API_URL}
                  onProfileClick={() => setShowProfile(true)}
                />
              </div>
            </div>
          </div>

          {/* MESSAGES AREA */}
          <div className="flex-1 overflow-y-auto px-0 py-8 bg-gradient-to-b from-zinc-50 to-white dark:from-zinc-950 dark:to-zinc-900">
            <MessageList
              messages={messages}
              messagesEndRef={messagesEndRef}
              darkMode={darkMode}
            />
          </div>

          {/* INPUT AREA */}
          <ChatInput
            onSendMessage={handleSendMessage}
            isLoading={isLoading}
            onFileUpload={handleFileUpload}
            isUploading={isUploading}
            activeThread={activeThread}
            onStopResponse={handleStopResponse}
          />
        </div>
      </div>
    </ErrorBoundary>
  );
};

export default ChatFrontend;
