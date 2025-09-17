import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { Send, Upload, FileText, User, Bot, Settings, MessageCircle, Plus, LogOut, Moon, Sun, AlertCircle, Loader2, X, Paperclip } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import remarkGfm from 'remark-gfm';
import LiquidEther from './LiquidEther';
import { List } from 'react-window'; // FIX: Use List, not FixedSizeList

// Configuration
//const AUTH_API_URL = 'http://36.255.14.28:9095';
//const API_BASE_URL = 'http://36.255.14.28:9096';
const AUTH_API_URL = 'http://192.168.190.28:9095';
const API_BASE_URL = 'http://192.168.190.28:9096';

// Utility functions with proper error handling
const createInitialThread = async (userEmail) => {
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

const loadUserThreads = async (userEmail) => {
  if (!userEmail || userEmail === 'undefined' || userEmail.trim() === '') {
    return [];
  }

  try {
    const response = await fetch(`${API_BASE_URL}/chat/threads?email=${encodeURIComponent(userEmail.trim())}`);
    
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

// Error boundary component
const ErrorBoundary = ({ children, fallback }) => {
  const [hasError, setHasError] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const handleError = (error) => {
      setHasError(true);
      setError(error);
    };

    window.addEventListener('error', handleError);
    return () => window.removeEventListener('error', handleError);
  }, []);

  if (hasError) {
    return fallback || (
      <div className="flex items-center justify-center h-screen bg-red-50 dark:bg-red-900">
        <div className="text-center">
          <AlertCircle className="w-16 h-16 text-red-500 mx-auto mb-4" />
          <h2 className="text-xl font-bold text-red-700 dark:text-red-300 mb-2">Something went wrong</h2>
          <p className="text-red-600 dark:text-red-400">{error?.message || 'An unexpected error occurred'}</p>
          <button 
            onClick={() => window.location.reload()} 
            className="mt-4 px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
          >
            Reload Page
          </button>
        </div>
      </div>
    );
  }

  return children;
};

// Toast notification component
const Toast = React.memo(({ message, type = 'error', onClose }) => {
  useEffect(() => {
    const timer = setTimeout(onClose, 5000);
    return () => clearTimeout(timer);
  }, [onClose]);

  const bgColor = type === 'error' ? 'bg-red-500' : 'bg-green-500';

  return (
    <div className={`fixed top-4 right-4 ${bgColor} text-white px-6 py-3 rounded-lg shadow-lg z-50 max-w-md`}>
      <div className="flex items-center justify-between">
        <span>{message}</span>
        <button onClick={onClose} className="ml-4 text-white hover:text-gray-200">×</button>
      </div>
    </div>
  );
});

const ChatFrontend = () => {
  // Core state
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [threads, setThreads] = useState([]);
  const [activeThread, setActiveThread] = useState(null);
  
  // User state with better initialization
  const [userEmail, setUserEmail] = useState('');
  const [userInfo, setUserInfo] = useState(null);
  const [isAuthLoading, setIsAuthLoading] = useState(true);
  const [authError, setAuthError] = useState(null);
  
  // UI state
  const [selectedModel, setSelectedModel] = useState('llama');
  
  // Thread-specific document storage
  const [threadDocuments, setThreadDocuments] = useState(new Map());
  
  const [showSettings, setShowSettings] = useState(false);
  const [darkMode, setDarkMode] = useState(() => localStorage.getItem('darkMode') === 'true');
  const [toast, setToast] = useState(null);
  const [initializationError, setInitializationError] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  
  // Email authentication states
  const [showEmailLogin, setShowEmailLogin] = useState(false);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isRegister, setIsRegister] = useState(false);
  const [emailAuthLoading, setEmailAuthLoading] = useState(false);
  
  // Refs
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  // PERFORMANCE FIX: Memoize current thread's document
  const getCurrentThreadDocument = useMemo(() => {
    return activeThread ? threadDocuments.get(activeThread) : null;
  }, [activeThread, threadDocuments]);

  // Set document for current thread
  const setCurrentThreadDocument = useCallback((documentInfo) => {
    if (activeThread) {
      setThreadDocuments(prev => {
        const newMap = new Map(prev);
        if (documentInfo) {
          newMap.set(activeThread, documentInfo);
        } else {
          newMap.delete(activeThread);
        }
        return newMap;
      });
    }
  }, [activeThread]);

  // PERFORMANCE FIX: Memoize utility functions to prevent re-renders
  const showToast = useCallback((message, type = 'error') => {
    setToast({ message, type });
  }, []);

  const closeToast = useCallback(() => {
    setToast(null);
  }, []);

  // PERFORMANCE FIX: Debounced scroll to bottom to prevent excessive calls
  const scrollToBottom = useCallback(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, []);

  const addSystemMessage = useCallback((content) => {
    const systemMsg = {
      role: 'system',
      content,
      timestamp: new Date().toISOString(),
      id: Date.now() + Math.random() // Add unique ID to prevent key issues
    };
    setMessages(prev => [...prev, systemMsg]);
  }, []);

  // Authentication effect - runs once on mount
  useEffect(() => {
    let isMounted = true; // Prevent state updates if component unmounted

    const fetchUserInfo = async () => {
      setIsAuthLoading(true);
      setAuthError(null);
      
      try {
        const response = await fetch(`${AUTH_API_URL}/api/user/me`, {
          credentials: 'include'
        });
        
        if (!isMounted) return;
        
        if (response.ok) {
          const data = await response.json();
          if (data && data.email && data.email !== 'undefined' && data.email.trim() !== '') {
            setUserInfo(data);
            setUserEmail(data.email.trim());
          } else {
            console.warn('Invalid user data received:', data);
            setAuthError('Invalid user data received from server');
          }
        } else if (response.status === 401) {
          setUserInfo(null);
          setUserEmail('');
        } else {
          setAuthError(`Authentication check failed: ${response.status}`);
        }
      } catch (error) {
        if (isMounted) {
          console.error('Error fetching user info:', error);
          setAuthError(`Failed to check authentication: ${error.message}`);
        }
      } finally {
        if (isMounted) {
          setIsAuthLoading(false);
        }
      }
    };
    
    fetchUserInfo();
    
    return () => {
      isMounted = false;
    };
  }, []);

  // Initialize user session when userEmail changes
  useEffect(() => {
    let isMounted = true;

    const initializeUserSession = async () => {
      if (!userEmail || userEmail === 'undefined') {
        return;
      }

      setInitializationError(null);
      
      try {
        const userThreads = await loadUserThreads(userEmail);
        if (!isMounted) return;

        setThreads(userThreads);

        if (userThreads && userThreads.length > 0) {
          const sortedThreads = userThreads.sort((a, b) => 
            new Date(b.created_at || b.timestamp || 0) - new Date(a.created_at || a.timestamp || 0)
          );
          
          const mostRecentThread = sortedThreads[0];
          setActiveThread(mostRecentThread.thread_id);
          await loadThreadHistory(mostRecentThread.thread_id);
          if (isMounted) {
            addSystemMessage(`Welcome back! Loaded your most recent chat: "${mostRecentThread.title}"`);
          }
        } else {
          const threadId = await createInitialThread(userEmail);
          if (isMounted) {
            setActiveThread(threadId);
            setMessages([]);
            addSystemMessage(`Welcome! Created your first chat session!`);
          }
        }
        
        // Clear thread documents when initializing
        if (isMounted) {
          setThreadDocuments(new Map());
        }

      } catch (error) {
        if (isMounted) {
          console.error('Failed to initialize user session:', error);
          setInitializationError(error.message);
          showToast(`Failed to initialize chat: ${error.message}`);
        }
      }
    };

    initializeUserSession();

    return () => {
      isMounted = false;
    };
  }, [userEmail]);

  // PERFORMANCE FIX: Debounced scroll effect
  useEffect(() => {
    const timer = setTimeout(() => {
      scrollToBottom();
    }, 100);

    return () => clearTimeout(timer);
  }, [messages.length]); // Only depend on message count, not entire messages array

  // Dark mode effect
  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
      localStorage.setItem('darkMode', 'true');
    } else {
      document.documentElement.classList.remove('dark');
      localStorage.setItem('darkMode', 'false');
    }
  }, [darkMode]);

  // Load thread history
  const loadThreadHistory = useCallback(async (threadId) => {
    if (!threadId) return;

    try {
      const response = await fetch(`${API_BASE_URL}/chat/history?thread_id=${threadId}`);
      
      if (!response.ok) {
        throw new Error(`Failed to load history: ${response.status}`);
      }
      
      const data = await response.json();
      
      let formattedMessages = [];
      if (Array.isArray(data)) {
        formattedMessages = data.map((msg, index) => ({
          role: msg.role,
          content: msg.content,
          timestamp: msg.created_at,
          id: msg.id || `${msg.created_at}-${index}` // Ensure unique IDs
        }));
      }
      
      setMessages(formattedMessages);
    } catch (error) {
      console.error('Error loading history:', error);
      showToast(`Failed to load chat history: ${error.message}`);
    }
  }, [showToast]);

  // Create new thread
  const createNewThread = useCallback(async () => {
    if (!userEmail) {
      showToast('Please log in to create a new chat');
      return;
    }

    try {
      const threadId = await createInitialThread(userEmail);
      setActiveThread(threadId);
      setMessages([]);
      
      const updatedThreads = await loadUserThreads(userEmail);
      setThreads(updatedThreads);
      
      addSystemMessage(`New chat session started!`);
    } catch (error) {
      console.error('Error creating new thread:', error);
      showToast(`Failed to create new chat: ${error.message}`);
    }
  }, [userEmail, showToast, addSystemMessage]);

  // File upload handler
  const handleFileUpload = useCallback(async (file) => {
    if (!activeThread) {
      showToast('Please start a conversation first');
      return;
    }

    if (!file) {
      showToast('No file selected');
      return;
    }

    const maxSize = 10 * 1024 * 1024; // 10MB
    if (file.size > maxSize) {
      showToast('File size must be less than 10MB');
      return;
    }

    const allowedTypes = ['application/pdf', 'text/plain', 'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'];
    if (!allowedTypes.includes(file.type)) {
      showToast('Supported file types: PDF, TXT, DOC, DOCX');
      return;
    }

    const formData = new FormData();
    formData.append('thread_id', activeThread);
    formData.append('file', file);

    try {
      setIsUploading(true);
      addSystemMessage(`Uploading "${file.name}"...`);
      
      const response = await fetch(`${API_BASE_URL}/chat/upload_doc`, {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Upload failed: ${response.status}`);
      }
      
      const data = await response.json();
      
      if (data.document_id) {
        const documentInfo = {
          id: data.document_id,
          filename: data.filename,
          chunks: data.chunks
        };
        
        setCurrentThreadDocument(documentInfo);
        
        addSystemMessage(`Document "${data.filename}" uploaded successfully. ${data.chunks} chunks processed. You can now ask questions about this document!`);
        showToast('Document uploaded successfully!', 'success');
      }
    } catch (error) {
      console.error('Error uploading file:', error);
      addSystemMessage(`Failed to upload document: ${error.message}`);
      showToast(`Upload failed: ${error.message}`);
    } finally {
      setIsUploading(false);
    }
  }, [activeThread, showToast, addSystemMessage, setCurrentThreadDocument]);

  // PERFORMANCE FIX: Optimized send message handler
  const sendMessage = useCallback(async (messageContent) => {
    if (!messageContent || !activeThread || isLoading) return;

    const messageId = Date.now() + Math.random();
    const userMessage = {
      role: 'human',
      content: messageContent,
      timestamp: new Date().toISOString(),
      id: messageId
    };

    setIsLoading(true);
    setMessages(prev => [...prev, userMessage]);

    try {
      let endpoint, payload;
      
      const currentDoc = getCurrentThreadDocument;
      
      if (currentDoc) {
        endpoint = `${API_BASE_URL}/chat/rag`;
        payload = {
          thread_id: activeThread,
          document_id: currentDoc.id,
          query: messageContent,
          model: selectedModel
        };
      } else {
        endpoint = `${API_BASE_URL}/chat/send`;
        payload = {
          thread_id: activeThread,
          message: messageContent,
          model: selectedModel
        };
      }

      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Request failed: ${response.status}`);
      }

      if (!response.body) {
        throw new Error('No response body received');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let assistantContent = '';

      const assistantMessageId = Date.now() + Math.random() + 1;
      const assistantMessage = {
        role: 'ai',
        content: '',
        timestamp: new Date().toISOString(),
        streaming: true,
        id: assistantMessageId
      };

      // Add assistant message placeholder
      setMessages(prev => [...prev, assistantMessage]);

      // PERFORMANCE FIX: Batch content updates to reduce re-renders
      let updateBuffer = '';
      let lastUpdate = Date.now();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (data === '[DONE]') continue;

            try {
              const parsed = JSON.parse(data);
              if (parsed.token) {
                assistantContent += parsed.token;
                updateBuffer += parsed.token;
                
                // PERFORMANCE FIX: Batch updates every 50ms to reduce re-renders
                const now = Date.now();
               
                  setMessages(prev => prev.map(msg => 
                    msg.id === assistantMessageId 
                      ? { ...msg, content: assistantContent }
                      : msg
                  ));
                  updateBuffer = '';
                  lastUpdate = now;
                
              } else if (parsed.error) {
                throw new Error(parsed.error);
              }
            } catch (parseError) {
              if (!data.includes('[DONE]')) {
                console.warn('Failed to parse streaming chunk:', data);
              }
            }
          }
        }
      }

      // Final update
      setMessages(prev => prev.map(msg => 
        msg.id === assistantMessageId 
          ? { ...msg, content: assistantContent, streaming: false }
          : msg
      ));

    } catch (error) {
      console.error('Error sending message:', error);
      addSystemMessage(`Error: ${error.message}`);
      showToast(`Failed to send message: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  }, [activeThread, isLoading, getCurrentThreadDocument, selectedModel, showToast, addSystemMessage]);

  // PERFORMANCE FIX: Optimized keyboard handler
  const handleKeyDown = useCallback((e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }, [sendMessage]);

  // Thread switching handler
  const handleThreadSwitch = useCallback(async (threadId) => {
    if (threadId === activeThread) return; // Prevent unnecessary switches
    
    setActiveThread(threadId);
    await loadThreadHistory(threadId);
  }, [loadThreadHistory, activeThread]);

  // Handler for email login/register
  const handleEmailAuth = useCallback(async (e) => {
    e.preventDefault();
    setEmailAuthLoading(true);
    const endpoint = isRegister ? `${AUTH_API_URL}/api/register` : `${AUTH_API_URL}/api/login`;
    try {
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ email, password, name: email.split('@')[0] })
      });
      const data = await res.json();
      if (res.ok) {
        window.location.reload();
      } else {
        showToast(data.message || 'Authentication failed');
      }
    } catch (err) {
      showToast('Network error');
    } finally {
      setEmailAuthLoading(false);
    }
  }, [isRegister, email, password, showToast]);

  // PERFORMANCE FIX: Memoize streaming indicator
  const StreamingDots = React.memo(() => (
    <span className="inline-flex items-center ml-2">
      <span className="w-2 h-2 bg-blue-400 rounded-full animate-bounce"></span>
      <span className="w-2 h-2 bg-blue-400 rounded-full animate-bounce mx-1" style={{ animationDelay: '0.15s' }}></span>
      <span className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0.3s' }}></span>
    </span>
  ));

  // 1. Move markdown components outside render cycle for stability
  const MARKDOWN_COMPONENTS = {
    code({ node, inline, className, children, ...props }) {
      const language = /language-(\w+)/.exec(className || "")?.[1];
      return inline ? (
        <code className="bg-gray-100 dark:bg-gray-800 rounded px-1 font-mono text-[0.95em]" {...props}>
          {children}
        </code>
      ) : (
        <SyntaxHighlighter
          style={oneDark}
          language={language || 'text'}
          PreTag="div"
          customStyle={{
            borderRadius: "0.75rem",
            fontSize: "1em",
            padding: "1.2em",
            margin: 0,
            background: "#18181b",
            border: "1px solid #374151",
            userSelect: "text",
          }}
          codeTagProps={{
            style: {
              background: "transparent",
              userSelect: "text",
            }
          }}
        >
          {String(children).replace(/\n$/, "")}
        </SyntaxHighlighter>
      );
    },
    p({ children, ...props }) {
      return <p className="mb-3 last:mb-0" {...props}>{children}</p>;
    },
  };

  // 2. Memoized MessageBubble with strict comparison
  const MessageBubble = React.memo(({ message }) => {
    const isUser = message.role === "human";
    const isSystem = message.role === "system";
    const isAI = message.role === "ai";

    const renderContent = useMemo(() => (
      <div style={{ userSelect: "text" }}>
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={MARKDOWN_COMPONENTS}
        >
          {message.content}
        </ReactMarkdown>
      </div>
    ), [message.content]);

    if (isAI) {
      return (
        <div className="flex w-full mb-4 justify-start">
          <div className="max-w-3xl w-full">
            <div className="text-base text-gray-900 dark:text-gray-100 transition-all">
              {renderContent}
              {message.streaming && (
                <span className="inline-flex items-center ml-2">
                  <span className="w-2 h-2 bg-blue-400 rounded-full animate-bounce"></span>
                  <span className="w-2 h-2 bg-blue-400 rounded-full animate-bounce mx-1" style={{ animationDelay: '0.15s' }}></span>
                  <span className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0.3s' }}></span>
                </span>
              )}
            </div>
            <div className="text-xs mt-2 text-gray-400 dark:text-gray-500 pl-2">
              {message.timestamp && new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            </div>
          </div>
        </div>
      );
    }

    if (isUser) {
      return (
        <div className="flex w-full mb-4 justify-end">
          <div className="max-w-3xl w-full flex justify-end">
            <div className="bg-gray-200 dark:bg-zinc-800 text-black dark:text-white rounded-3xl px-7 py-5 shadow-md text-base transition-all">
              {renderContent}
            </div>
          </div>
        </div>
      );
    }

    return (
      <div className="flex w-full mb-4 justify-center">
        <div className="message-bubble bg-gray-200/60 dark:bg-zinc-800/60 text-gray-800 dark:text-gray-100 border border-gray-300 dark:border-zinc-700 rounded-2xl px-6 py-3 text-sm shadow"
          style={{ backdropFilter: 'blur(2px)' }}>
          {renderContent}
        </div>
      </div>
    );
  }, (prevProps, nextProps) =>
    prevProps.message.id === nextProps.message.id &&
    prevProps.message.content === nextProps.message.content &&
    prevProps.message.streaming === nextProps.message.streaming &&
    prevProps.message.timestamp === nextProps.message.timestamp
  );

  // 3. Memoized MessageList, virtualized if > 50 messages
  const MessageList = React.memo(({ messages, messagesEndRef }) => {
    const Row = useCallback(({ index, style }) => (
      <div style={style}>
        <MessageBubble message={messages[index]} />
      </div>
    ), [messages]);

    if (messages.length > 50) {
      return (
        <List
          height={600}
          itemCount={messages.length}
          itemSize={120}
          width="100%"
        >
          {Row}
        </List>
      );
    }

    return (
      <div className="max-w-4xl mx-auto px-2">
        {messages.map((message) => (
          <MessageBubble key={message.id} message={message} />
        ))}
        <div ref={messagesEndRef} />
      </div>
    );
  }, (prevProps, nextProps) => prevProps.messages === nextProps.messages);

  // 4. Memoized ChatInput with isolated state
  const ChatInput = React.memo(({ onSendMessage, isLoading, onFileUpload, isUploading, activeThread }) => {
    const [inputValue, setInputValue] = useState('');
    const fileInputRef = useRef(null);

    const handleInputChange = useCallback((e) => {
      setInputValue(e.target.value);
      e.target.style.height = 'auto';
      e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px';
    }, []);

    const handleSend = useCallback(() => {
      if (inputValue.trim() && !isLoading) {
        onSendMessage(inputValue.trim());
        setInputValue('');
      }
    }, [inputValue, isLoading, onSendMessage]);

    const handleKeyDown = useCallback((e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    }, [handleSend]);

    return (
      <div className="p-6">
        <div className="max-w-3xl mx-auto">
          <div className="relative flex items-end bg-white dark:bg-zinc-900 rounded-2xl shadow-md border border-zinc-200 dark:border-zinc-800 overflow-hidden">
            <div className="p-2">
              <input
                ref={fileInputRef}
                type="file"
                onChange={(e) => e.target.files[0] && onFileUpload(e.target.files[0])}
                className="hidden"
                accept=".pdf,.txt,.doc,.docx"
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                disabled={!activeThread || isUploading}
                className="p-2 rounded-full hover:bg-gray-100 dark:hover:bg-zinc-800 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                title="Upload document (PDF, TXT, DOC, DOCX)"
              >
                {isUploading ? (
                  <Loader2 className="w-5 h-5 animate-spin text-gray-500" />
                ) : (
                  <Paperclip className="w-5 h-5 text-gray-500 dark:text-gray-400" />
                )}
              </button>
            </div>
            <textarea
              value={inputValue}
              onChange={handleInputChange}
              onKeyDown={handleKeyDown}
              placeholder="Ask anything..."
              disabled={isLoading}
              className="bg-transparent resize-none focus:outline-none text-base text-zinc-900 dark:text-zinc-100 placeholder-zinc-400 dark:placeholder-zinc-500 py-3 min-h-[48px] max-w-full w-full disabled:opacity-50 disabled:cursor-not-allowed"
              rows={1}
              style={{ maxHeight: '120px' }}
            />
            <div className="p-2">
              <button
                onClick={handleSend}
                disabled={!inputValue.trim() || isLoading}
                className="bg-gradient-to-r from-blue-500 to-purple-600 text-white p-2 rounded-full shadow-lg hover:from-blue-600 hover:to-purple-700 transition-all duration-200 flex items-center justify-center disabled:opacity-50 disabled:cursor-not-allowed w-10 h-10"
              >
                {isLoading ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Send className="w-4 h-4" />
                )}
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  });

  // PERFORMANCE FIX: Memoize user circle component
  const UserCircle = React.memo(() => {
    const [showLogout, setShowLogout] = useState(false);

    useEffect(() => {
      const handleClick = (e) => {
        if (!e.target.closest('.user-circle-dropdown')) setShowLogout(false);
      };
      if (showLogout) document.addEventListener('mousedown', handleClick);
      return () => document.removeEventListener('mousedown', handleClick);
    }, [showLogout]);

    return (
      <div className="relative user-circle-dropdown">
        <button
          className="w-10 h-10 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center shadow-lg border-2 border-white hover:scale-105 transition-transform focus:outline-none"
          title={userInfo?.name || userEmail}
          onClick={() => setShowLogout(v => !v)}
        >
          {userInfo?.picture ? (
            <img
              src={userInfo.picture}
              alt={userInfo.name}
              className="w-9 h-9 rounded-full object-cover"
            />
          ) : (
            <User className="w-6 h-6 text-white" />
          )}
        </button>
        {showLogout && (
          <div className="absolute right-0 mt-2 bg-white dark:bg-zinc-800 border border-gray-200 dark:border-zinc-700 shadow-lg rounded-lg z-50 min-w-[120px]">
            {(userInfo?.name || userEmail) && (
              <div className="px-4 py-2 text-xs text-gray-700 dark:text-gray-300 border-b dark:border-zinc-700">
                {userInfo?.name ? userInfo.name : userEmail}
              </div>
            )}
            <a
              href={`${AUTH_API_URL}/logout`}
              className="flex items-center gap-2 px-4 py-2 text-gray-700 dark:text-gray-300 hover:bg-red-50 dark:hover:bg-red-900 hover:text-red-600 dark:hover:text-red-400 rounded-b-lg transition"
            >
              <LogOut className="w-4 h-4" />
              Logout
            </a>
          </div>
        )}
      </div>
    );
  });

  // PERFORMANCE FIX: Memoize dark mode toggle
  const DarkModeToggle = React.memo(() => (
    <button
      onClick={() => setDarkMode(v => !v)}
      className="p-2 rounded-full hover:bg-gray-200 dark:hover:bg-gray-700 transition"
      title={darkMode ? "Switch to Light Mode" : "Switch to Dark Mode"}
    >
      {darkMode ? <Sun className="w-5 h-5 text-yellow-400" /> : <Moon className="w-5 h-5 text-gray-700" />}
    </button>
  ));

  // Loading screen
  if (isAuthLoading) {
    return (
      <div className="flex flex-col items-center justify-center h-screen bg-gradient-to-br from-gray-50 to-gray-100">
        <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mb-4"></div>
        <p className="text-gray-600">Checking authentication...</p>
      </div>
    );
  }

  // Login screen
  if (!userEmail) {
    return (
      <div className="relative w-screen h-screen overflow-hidden">
        {/* LiquidEther animated background */}
        <div className="absolute inset-0 w-full h-full" style={{ zIndex: 1, background: "#060010" }}>
          <LiquidEther
            colors={['#5227ff', '#9eefff', '#b19eef']}
            mouseForce={20}
            cursorSize={100}
            isViscous={false}
            viscous={30}
            iterationsViscous={32}
            iterationsPoisson={32}
            resolution={0.5}
            isBounce={false}
            autoDemo={false}
            autoSpeed={0.5}
            autoIntensity={2.2}
            takeoverDuration={0.25}
            autoResumeDelay={3000}
            autoRampDuration={0.6}
          />
        </div>

        {/* Login content overlay */}
        <div 
          className="absolute inset-0 flex flex-col items-center justify-center"
          style={{ zIndex: 10, pointerEvents: 'none' }}
        >
          <div
            className="bg-white/10 backdrop-blur-xl shadow-2xl flex flex-col items-center max-w-md w-full mx-4"
            style={{
              height: '500px',
              width: '400px',
              borderRadius: '1.5rem',
              border: '1px solid rgba(255, 255, 255, 0.2)',
              boxShadow: '0 15px 30px 0 rgba(6, 12, 70, 0.37)',
              padding: '2rem',
              pointerEvents: 'all',
            }}
          >
            <img src="/nmit-logoo.png" alt="NMIT Logo" className="w-100 h-100 mb-4" />
            <p className="mb-6 text-white/80 text-center">Welcome to LLMNet</p>
            
            {authError && (
              <div className="mb-4 p-3 bg-red-500/20 border border-red-400/30 rounded-lg text-red-200 text-sm backdrop-blur-sm">
                <AlertCircle className="w-4 h-4 inline mr-2" />
                {authError}
              </div>
            )}

            {!showEmailLogin ? (
              <>
                <a
                  href={`${AUTH_API_URL}/login`}
                  className="bg-blue-600/90 hover:bg-blue-700/90 text-white px-6 py-3 rounded-lg font-medium flex items-center transition-colors mb-4 backdrop-blur-sm"
                >
                  <svg className="w-5 h-5 mr-2" viewBox="0 0 24 24">
                    <path fill="currentColor" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                    <path fill="currentColor" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                    <path fill="currentColor" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                    <path fill="currentColor" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
                  </svg>
                  Sign in with Google
                </a>
                <button
                  className="text-white/80 hover:text-white underline text-sm transition-colors"
                  onClick={() => setShowEmailLogin(true)}
                >
                  Or sign in with Email
                </button>
                <p className="mt-6 text-xs text-white/60 text-center">
                  Department of ISE, NMIT
                </p>
              </>
            ) : (
              <form className="w-full" onSubmit={handleEmailAuth}>
                <input
                  type="email"
                  className="w-full mb-2 px-4 py-2 border border-white/20 rounded-lg bg-white/10 backdrop-blur-sm text-white placeholder-white/60"
                  placeholder="Email"
                  value={email}
                  onChange={e => setEmail(e.target.value)}
                  required
                />
                <input
                  type="password"
                  className="w-full mb-2 px-4 py-2 border border-white/20 rounded-lg bg-white/10 backdrop-blur-sm text-white placeholder-white/60"
                  placeholder="Password"
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  required
                />
                <button
                  type="submit"
                  className="w-full bg-blue-600/90 hover:bg-blue-700/90 text-white px-4 py-2 rounded-lg font-medium mb-2 backdrop-blur-sm transition-colors"
                  disabled={emailAuthLoading}
                >
                  {emailAuthLoading ? (
                    <Loader2 className="w-4 h-4 animate-spin mx-auto" />
                  ) : (
                    isRegister ? 'Register' : 'Login'
                  )}
                </button>
                <button
                  type="button"
                  className="w-full text-sm text-white/80 hover:text-white underline transition-colors"
                  onClick={() => setIsRegister(r => !r)}
                >
                  {isRegister ? 'Already have an account? Login' : "Don't have an account? Register"}
                </button>
                <button
                  type="button"
                  className="w-full text-xs text-white/60 hover:text-white/80 mt-2 transition-colors"
                  onClick={() => setShowEmailLogin(false)}
                >
                  Back to Google Login
                </button>
              </form>
            )}
          </div>
        </div>
      </div>
    );
  }

  // Error screen for initialization issues
  if (initializationError) {
    return (
      <div className="flex flex-col items-center justify-center h-screen bg-gradient-to-br from-red-50 to-red-100">
        <div className="bg-white p-8 rounded-lg shadow-lg flex flex-col items-center max-w-md">
          <AlertCircle className="w-16 h-16 text-red-500 mb-4" />
          <h2 className="text-2xl font-bold mb-2 text-red-700">Initialization Failed</h2>
          <p className="mb-6 text-gray-600 text-center">{initializationError}</p>
          <button 
            onClick={() => window.location.reload()} 
            className="bg-red-600 hover:bg-red-700 text-white px-6 py-3 rounded-lg font-medium transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  // Main Chat Interface
  return (
    <ErrorBoundary>
      <div className="flex h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-zinc-900 dark:to-zinc-950 relative">
        {/* Toast notifications */}
        {toast && (
          <Toast 
            message={toast.message} 
            type={toast.type} 
            onClose={closeToast} 
          />
        )}

        {/* Sidebar */}
        <div className="w-80 bg-white dark:bg-zinc-900 shadow-2xl border-r border-zinc-200 dark:border-zinc-800 flex flex-col">
          <div className="p-7 border-b border-zinc-200 dark:border-zinc-800 flex items-center">
            <h1
              className={`text-3xl font-extrabold tracking-tight ${
                darkMode ? "text-white" : "text-black"
              }`}
            >
              LLMNet
            </h1>
          </div>

          {/* New Chat Button */}
          <div className="p-5">
            <button
              onClick={createNewThread}
              disabled={isLoading}
              className={`w-full py-3 px-4 rounded-xl font-semibold transition-all shadow hover:shadow-md flex items-center justify-center disabled:opacity-50 disabled:cursor-not-allowed
                ${darkMode
                  ? "bg-zinc-800 text-white hover:bg-zinc-700"
                  : "bg-zinc-100 text-black hover:bg-zinc-200"
                }`
              }
            >
              <Plus className="w-5 h-5 mr-2" />
              New Chat
            </button>
          </div>

          {/* Document Status */}
          {getCurrentThreadDocument && (
            <div className="mx-5 mb-5">
              <div className="bg-green-50 dark:bg-green-900 rounded-lg border border-green-200 dark:border-green-700 p-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center text-green-800 dark:text-green-200">
                    <FileText className="w-4 h-4 mr-2" />
                    <span className="text-sm font-medium truncate">{getCurrentThreadDocument.filename}</span>
                  </div>
                  <button
                    onClick={() => setCurrentThreadDocument(null)}
                    className="text-green-600 dark:text-green-300 hover:text-green-800 dark:hover:text-green-100"
                    title="Remove document"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
                <div className="text-xs text-green-600 dark:text-green-300 mt-1">
                  {getCurrentThreadDocument.chunks} chunks processed
                </div>
              </div>
            </div>
          )}

          {/* Settings */}
          <div className="px-5 pb-5">
            <button
              onClick={() => setShowSettings(!showSettings)}
              className="w-full bg-zinc-100 dark:bg-zinc-800 text-zinc-700 dark:text-zinc-200 py-2 px-4 rounded-xl font-semibold hover:bg-zinc-200 dark:hover:bg-zinc-700 transition-all flex items-center justify-center"
            >
              <Settings className="w-4 h-4 mr-2" />
              Settings
            </button>
            {showSettings && (
              <div className="mt-3 p-3 bg-zinc-50 dark:bg-zinc-900 rounded-lg border border-zinc-200 dark:border-zinc-700 space-y-3">
                <div>
                  <label className="block text-xs font-semibold text-zinc-700 dark:text-zinc-300 mb-1">Model</label>
                  <select
                    value={selectedModel}
                    onChange={(e) => setSelectedModel(e.target.value)}
                    className="w-full p-2 border border-zinc-300 dark:border-zinc-700 rounded-md text-base bg-white dark:bg-zinc-900 text-zinc-900 dark:text-zinc-100"
                  >
                    <option value="llama">Llama 3.1 8B</option>
                    <option value="deepseek">DeepSeek Coder V2</option>
                  </select>
                </div>
              </div>
            )}
          </div>

          {/* Thread List */}
          <div className="flex-1 overflow-y-auto px-5">
            <h3 className="text-base font-bold text-zinc-700 dark:text-zinc-200 mb-3">Recent Conversations</h3>
            <div className="space-y-2">
              {threads.map((thread) => (
                <button
                  key={thread.thread_id}
                  onClick={() => handleThreadSwitch(thread.thread_id)}
                  className={`w-full text-left p-3 rounded-xl transition-all
                    ${activeThread === thread.thread_id
                      ? 'bg-gray-100 dark:bg-zinc-800 border border-gray-300 dark:border-zinc-700'
                      : 'bg-white dark:bg-zinc-900 border border-zinc-100 dark:border-zinc-700'
                    }
                    hover:bg-gray-100 dark:hover:bg-zinc-800`}
                >
                  <div className="text-base font-semibold text-zinc-800 dark:text-zinc-100 truncate">{thread.title}</div>
                  <div className="text-xs text-zinc-500 dark:text-zinc-400 mt-1">
                    {new Date(thread.created_at || thread.timestamp).toLocaleDateString()}
                  </div>
                  {threadDocuments.get(thread.thread_id) && (
                    <div className="flex items-center mt-1">
                      <FileText className="w-3 h-3 text-green-600 dark:text-green-400 mr-1" />
                      <span className="text-xs text-green-600 dark:text-green-400 truncate">
                        {threadDocuments.get(thread.thread_id).filename}
                      </span>
                    </div>
                  )}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Main Chat Area */}
        <div className="flex-1 flex flex-col bg-gradient-to-br from-white to-zinc-50 dark:from-zinc-950 dark:to-zinc-900">
          {/* Header */}
          <div className="bg-white dark:bg-zinc-900 shadow-sm border-b border-zinc-200 dark:border-zinc-800 p-6">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-2xl font-bold text-zinc-800 dark:text-zinc-100">
                  AI Chat
                </h2>
                <p className="text-base text-zinc-600 dark:text-zinc-400">
                  {getCurrentThreadDocument 
                    ? `Chatting about: ${getCurrentThreadDocument.filename}`
                    : 'General conversation with AI'
                  }
                </p>
              </div>
              <div className="flex items-center gap-5">
                <DarkModeToggle />
                <div className="text-base text-zinc-500 dark:text-zinc-300 whitespace-nowrap">
                  Model: {selectedModel === 'llama' ? 'Llama 3.1 8B' : 'DeepSeek Coder V2'}
                </div>
                <UserCircle />
              </div>
            </div>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto px-0 py-8 bg-gradient-to-b from-zinc-50 to-white dark:from-zinc-950 dark:to-zinc-900">
            <MessageList messages={messages} messagesEndRef={messagesEndRef} />
          </div>

          {/* Input Area */}
          <ChatInput
            onSendMessage={sendMessage}
            isLoading={isLoading}
            onFileUpload={handleFileUpload}
            isUploading={isUploading}
            activeThread={activeThread}
          />
        </div>
      </div>
    </ErrorBoundary>
  );
};

export default ChatFrontend;