import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Send, Upload, FileText, User, Bot, Settings, MessageCircle, Plus, LogOut, Moon, Sun, AlertCircle } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import remarkGfm from 'remark-gfm';

// Configuration
const AUTH_API_URL = 'http://localhost:9095';
const API_BASE_URL = 'http://172.17.35.82:9096';

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
const Toast = ({ message, type = 'error', onClose }) => {
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
};

const ChatFrontend = () => {
  // Core state
  const [activeMode, setActiveMode] = useState('chat');
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
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
  const [uploadedDoc, setUploadedDoc] = useState(null);
  const [showSettings, setShowSettings] = useState(false);
  const [darkMode, setDarkMode] = useState(() => localStorage.getItem('darkMode') === 'true');
  const [toast, setToast] = useState(null);
  const [initializationError, setInitializationError] = useState(null);
  
  // Refs
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  // Utility functions
  const showToast = useCallback((message, type = 'error') => {
    setToast({ message, type });
  }, []);

  const closeToast = useCallback(() => {
    setToast(null);
  }, []);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  const addSystemMessage = useCallback((content) => {
    const systemMsg = {
      role: 'system',
      content,
      timestamp: new Date().toISOString()
    };
    setMessages(prev => [...prev, systemMsg]);
  }, []);

  // Authentication effect - runs once on mount
  useEffect(() => {
    const fetchUserInfo = async () => {
      setIsAuthLoading(true);
      setAuthError(null);
      
      try {
        const response = await fetch(`${AUTH_API_URL}/api/user/me`, {
          credentials: 'include'
        });
        
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
          // User not authenticated - this is expected
          setUserInfo(null);
          setUserEmail('');
        } else {
          setAuthError(`Authentication check failed: ${response.status}`);
        }
      } catch (error) {
        console.error('Error fetching user info:', error);
        setAuthError(`Failed to check authentication: ${error.message}`);
      } finally {
        setIsAuthLoading(false);
      }
    };
    
    fetchUserInfo();
  }, []);

  // FIXED: Initialize user session when userEmail changes - load most recent chat instead of creating new one
  useEffect(() => {
    const initializeUserSession = async () => {
      if (!userEmail || userEmail === 'undefined') {
        return;
      }

      setInitializationError(null);
      
      try {
        // Load existing threads first
        const userThreads = await loadUserThreads(userEmail);
        setThreads(userThreads);

        // If user has existing threads, load the most recent one
        if (userThreads && userThreads.length > 0) {
          // Sort threads by creation date (most recent first)
          const sortedThreads = userThreads.sort((a, b) => 
            new Date(b.created_at || b.timestamp || 0) - new Date(a.created_at || a.timestamp || 0)
          );
          
          const mostRecentThread = sortedThreads[0];
          setActiveThread(mostRecentThread.thread_id);
          
          // Load the history of the most recent thread
          await loadThreadHistory(mostRecentThread.thread_id);
          
          addSystemMessage(`👋 Welcome back! Loaded your most recent chat: "${mostRecentThread.title}"`);
        } else {
          // Only create a new thread if user has no existing threads
          const threadId = await createInitialThread(userEmail);
          setActiveThread(threadId);
          setMessages([]);
          addSystemMessage(`🆕 Welcome! Created your first ${activeMode === 'rag' ? 'RAG' : 'Chat'} session!`);
        }
        
        // Clear uploaded document state when switching users
        setUploadedDoc(null);

      } catch (error) {
        console.error('Failed to initialize user session:', error);
        setInitializationError(error.message);
        showToast(`Failed to initialize chat: ${error.message}`);
      }
    };

    initializeUserSession();
  }, [userEmail, showToast, addSystemMessage]); // Removed activeMode dependency to prevent re-initialization

  // Scroll to bottom when messages change
  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

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
      const endpoint = activeMode === 'rag' ? 
        `${API_BASE_URL}/chat/rag_history?thread_id=${threadId}` :
        `${API_BASE_URL}/chat/history?thread_id=${threadId}`;
      
      const response = await fetch(endpoint);
      
      if (!response.ok) {
        throw new Error(`Failed to load history: ${response.status}`);
      }
      
      const data = await response.json();
      
      let formattedMessages = [];
      if (activeMode === 'rag' && data.messages) {
        formattedMessages = data.messages.map(msg => ({
          role: msg.role,
          content: msg.content,
          timestamp: msg.timestamp
        }));
      } else if (Array.isArray(data)) {
        formattedMessages = data.map(msg => ({
          role: msg.role,
          content: msg.content,
          timestamp: msg.created_at
        }));
      }
      
      setMessages(formattedMessages);
    } catch (error) {
      console.error('Error loading history:', error);
      showToast(`Failed to load chat history: ${error.message}`);
    }
  }, [activeMode, showToast]);

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
      setUploadedDoc(null);
      
      // Reload threads list
      const updatedThreads = await loadUserThreads(userEmail);
      setThreads(updatedThreads);
      
      addSystemMessage(`🆕 New ${activeMode === 'rag' ? 'RAG' : 'Chat'} session started!`);
    } catch (error) {
      console.error('Error creating new thread:', error);
      showToast(`Failed to create new chat: ${error.message}`);
    }
  }, [userEmail, activeMode, showToast, addSystemMessage]);

  // File upload handler
  const handleFileUpload = useCallback(async (file) => {
    if (!activeThread || activeMode !== 'rag') {
      showToast('Please ensure you\'re in RAG mode with an active thread');
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
      setIsLoading(true);
      addSystemMessage(`📤 Uploading "${file.name}"...`);
      
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
        setUploadedDoc({
          id: data.document_id,
          filename: data.filename,
          chunks: data.chunks
        });
        
        addSystemMessage(`📄 Document "${data.filename}" uploaded successfully. ${data.chunks} chunks processed. Ready for Q&A!`);
        showToast('Document uploaded successfully!', 'success');
      }
    } catch (error) {
      console.error('Error uploading file:', error);
      addSystemMessage(`❌ Failed to upload document: ${error.message}`);
      showToast(`Upload failed: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  }, [activeThread, activeMode, showToast, addSystemMessage]);

  // Send message handler
  const sendMessage = useCallback(async () => {
    if (!inputMessage.trim() || !activeThread || isLoading) return;

    if (activeMode === 'rag' && !uploadedDoc) {
      showToast('Please upload a document first for RAG chat');
      return;
    }

    const userMessage = {
      role: 'human',
      content: inputMessage.trim(),
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMessage]);
    const currentMessage = inputMessage.trim();
    setInputMessage('');
    setIsLoading(true);

    try {
      let endpoint, payload;
      
      if (activeMode === 'rag' && uploadedDoc) {
        endpoint = `${API_BASE_URL}/chat/rag`;
        payload = {
          thread_id: activeThread,
          document_id: uploadedDoc.id,
          query: currentMessage,
          model: selectedModel
        };
      } else {
        endpoint = `${API_BASE_URL}/chat/send`;
        payload = {
          thread_id: activeThread,
          message: currentMessage,
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

      const assistantMessage = {
        role: 'ai',
        content: '',
        timestamp: new Date().toISOString(),
        streaming: true
      };

      setMessages(prev => [...prev, assistantMessage]);

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
                setMessages(prev => prev.map((msg, idx) => 
                  idx === prev.length - 1 
                    ? { ...msg, content: assistantContent }
                    : msg
                ));
              } else if (parsed.error) {
                throw new Error(parsed.error);
              }
            } catch (parseError) {
              // Ignore parsing errors for streaming chunks
              if (!data.includes('[DONE]')) {
                console.warn('Failed to parse streaming chunk:', data);
              }
            }
          }
        }
      }

      // Mark streaming as complete
      setMessages(prev => prev.map((msg, idx) => 
        idx === prev.length - 1 
          ? { ...msg, streaming: false }
          : msg
      ));

    } catch (error) {
      console.error('Error sending message:', error);
      addSystemMessage(`❌ Error: ${error.message}`);
      showToast(`Failed to send message: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  }, [inputMessage, activeThread, isLoading, activeMode, uploadedDoc, selectedModel, showToast, addSystemMessage]);

  // Keyboard handler for message input
  const handleKeyDown = useCallback((e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }, [sendMessage]);

  // Format timestamp
  const formatTimestamp = useCallback((timestamp) => {
    try {
      return new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } catch {
      return '';
    }
  }, []);

  // Fixed MessageBubble component with proper code block separation
  const MessageBubble = React.memo(({ message }) => {
    const isUser = message.role === "human";
    const isSystem = message.role === "system";
    const isAI = message.role === "ai";

    const CodeBlock = ({ children, language }) => {
      const codeRef = useRef(null);
      const [copied, setCopied] = useState(false);

      const handleCopy = useCallback(() => {
        if (codeRef.current) {
          const textContent = codeRef.current.textContent || codeRef.current.innerText;
          navigator.clipboard.writeText(textContent);
          setCopied(true);
          setTimeout(() => setCopied(false), 1200);
        }
      }, []);

      return (
        <div className="relative group my-4">
          <button
            onClick={handleCopy}
            className="absolute top-2 right-2 z-10 bg-white/90 dark:bg-gray-800/90 hover:bg-blue-100 dark:hover:bg-blue-900 text-xs px-2 py-1 rounded border border-gray-200 dark:border-gray-700 shadow transition-all opacity-0 group-hover:opacity-100 focus:opacity-100"
            title="Copy code"
          >
            {copied ? "Copied!" : "Copy"}
          </button>
          <div ref={codeRef}>
            <SyntaxHighlighter
              style={oneDark}
              language={language || 'text'}
              PreTag="div"
              customStyle={{
                borderRadius: "0.5rem",
                fontSize: "0.9em",
                padding: "1em",
                margin: 0,
                background: "#1e1e1e",
                border: "1px solid #374151",
                userSelect: "text",
                WebkitUserSelect: "text",
                MozUserSelect: "text",
                msUserSelect: "text",
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
          </div>
        </div>
      );
    };

    const renderAIMarkdown = (content) => (
      <div style={{ userSelect: "text" }}>
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            code({ node, inline, className, children, ...props }) {
              const language = /language-(\w+)/.exec(className || "")?.[1];
              return inline ? (
                <code
                  className="bg-gray-100 dark:bg-gray-800 rounded px-1 font-mono text-[0.85em] text-pink-700 dark:text-pink-300"
                  style={{ userSelect: "text", background: "rgb(243 244 246)" }}
                  {...props}
                >
                  {children}
                </code>
              ) : (
                <CodeBlock language={language}>{children}</CodeBlock>
              );
            },
            p({ children, ...props }) {
              return (
                <p 
                  className="mb-3 last:mb-0" 
                  style={{ userSelect: "text", background: "transparent" }}
                  {...props}
                >
                  {children}
                </p>
              );
            },
          }}
        >
          {content}
        </ReactMarkdown>
      </div>
    );

    const renderUserMarkdown = (content) => (
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          code({ node, inline, className, children, ...props }) {
            const language = /language-(\w+)/.exec(className || "")?.[1];
            return inline ? (
              <code
                className="bg-blue-400/20 rounded px-1 font-mono text-[0.85em] text-blue-100"
                style={{ userSelect: "text" }}
                {...props}
              >
                {children}
              </code>
            ) : (
              <div className="my-3 bg-blue-600/20 rounded-lg p-3 border border-blue-400/30">
                <code 
                  className="text-blue-100 font-mono text-sm whitespace-pre-wrap block"
                  style={{ userSelect: "text", background: "transparent" }}
                >
                  {children}
                </code>
              </div>
            );
          },
        }}
      >
        {content}
      </ReactMarkdown>
    );

    const renderSystemMarkdown = (content) => (
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          code({ node, inline, className, children, ...props }) {
            const language = /language-(\w+)/.exec(className || "")?.[1];
            return inline ? (
              <code
                className="bg-orange-200 dark:bg-orange-800 rounded px-1 font-mono text-[0.85em] text-orange-800 dark:text-orange-200"
                style={{ userSelect: "text" }}
                {...props}
              >
                {children}
              </code>
            ) : (
              <div className="my-3 bg-orange-200 dark:bg-orange-800 rounded-lg p-3 border border-orange-300 dark:border-orange-700">
                <code 
                  className="text-orange-800 dark:text-orange-200 font-mono text-sm whitespace-pre-wrap block"
                  style={{ userSelect: "text", background: "transparent" }}
                >
                  {children}
                </code>
              </div>
            );
          },
        }}
      >
        {content}
      </ReactMarkdown>
    );

    if (isAI) {
      return (
        <div className="message-bubble flex mb-6 justify-start">
          <div className="flex-shrink-0 w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center mr-3">
            <Bot className="w-4 h-4 text-white" />
          </div>
          <div 
            className="flex-1 min-w-0 max-w-4xl"
            style={{ userSelect: "text" }}
          >
            <div 
              className="text-sm leading-relaxed text-gray-900 dark:text-gray-100"
              style={{ userSelect: "text", background: "transparent" }}
            >
              {renderAIMarkdown(message.content)}
            </div>
            {message.streaming && (
              <div className="inline-flex items-center mt-2">
                <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse"></div>
                <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse ml-1 delay-75"></div>
                <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse ml-1 delay-150"></div>
              </div>
            )}
            <div className="text-xs mt-2 text-gray-500 dark:text-gray-400">
              {formatTimestamp(message.timestamp)}
            </div>
          </div>
        </div>
      );
    }

    return (
      <div className={`message-bubble flex mb-6 ${isUser ? "justify-end" : "justify-start"}`}>
        {!isUser && !isSystem && (
          <div className="flex-shrink-0 w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center mr-3">
            <Bot className="w-4 h-4 text-white" />
          </div>
        )}

        <div
          className={`max-w-xs lg:max-w-md px-4 py-3 rounded-2xl relative ${
            isUser
              ? "bg-gradient-to-br from-blue-500 to-blue-600 text-white"
              : isSystem
              ? "bg-gradient-to-br from-orange-100 to-orange-200 text-orange-800 dark:bg-orange-900 dark:text-orange-100 border border-orange-300 dark:border-orange-700"
              : "bg-white dark:bg-gray-900 text-gray-800 dark:text-gray-100 shadow-lg border border-gray-100 dark:border-gray-700"
          }`}
          style={{ userSelect: "text" }}
        >
          <div 
            className="text-sm leading-relaxed"
            style={{ userSelect: "text", background: "transparent" }}
          >
            {isUser ? renderUserMarkdown(message.content) : 
             isSystem ? renderSystemMarkdown(message.content) : 
             renderAIMarkdown(message.content)}
          </div>
          <div
            className={`text-xs mt-2 ${
              isUser
                ? "text-blue-100"
                : isSystem
                ? "text-orange-600 dark:text-orange-200"
                : "text-gray-500 dark:text-gray-400"
            }`}
          >
            {formatTimestamp(message.timestamp)}
          </div>
        </div>

        {isUser && (
          <div className="flex-shrink-0 w-8 h-8 bg-gradient-to-br from-green-500 to-emerald-600 rounded-full flex items-center justify-center ml-3">
            <User className="w-4 h-4 text-white" />
          </div>
        )}
      </div>
    );
  });

  // User circle component
  const UserCircle = () => {
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
          <div className="absolute right-0 mt-2 bg-white border border-gray-200 shadow-lg rounded-lg z-50 min-w-[120px]">
            {(userInfo?.name || userEmail) && (
              <div className="px-4 py-2 text-xs text-gray-700 border-b">
                {userInfo?.name ? userInfo.name : userEmail}
              </div>
            )}
            <a
              href={`${AUTH_API_URL}/logout`}
              className="flex items-center gap-2 px-4 py-2 text-gray-700 hover:bg-red-50 hover:text-red-600 rounded-b-lg transition"
            >
              <LogOut className="w-4 h-4" />
              Logout
            </a>
          </div>
        )}
      </div>
    );
  };

  // Dark mode toggle
  const DarkModeToggle = () => (
    <button
      onClick={() => setDarkMode(v => !v)}
      className="p-2 rounded-full hover:bg-gray-200 dark:hover:bg-gray-700 transition"
      title={darkMode ? "Switch to Light Mode" : "Switch to Dark Mode"}
    >
      {darkMode ? <Sun className="w-5 h-5 text-yellow-400" /> : <Moon className="w-5 h-5 text-gray-700" />}
    </button>
  );

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
      <div className="flex flex-col items-center justify-center h-screen bg-gradient-to-br from-gray-50 to-gray-100">
        <div className="bg-white p-8 rounded-lg shadow-lg flex flex-col items-center max-w-md">
          <img src="https://upload.wikimedia.org/wikipedia/commons/5/53/Google_%22G%22_Logo.svg" alt="Google" className="w-12 h-12 mb-4" />
          <h2 className="text-2xl font-bold mb-2">Sign in to LLMNet</h2>
          <p className="mb-6 text-gray-600 text-center">Please sign in with your Google account to continue.</p>
          
          {authError && (
            <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
              <AlertCircle className="w-4 h-4 inline mr-2" />
              {authError}
            </div>
          )}
          
          <a
            href={`${AUTH_API_URL}/login`}
            className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg font-medium flex items-center transition-colors"
          >
            <svg className="w-5 h-5 mr-2" viewBox="0 0 48 48">
              <g>
                <path fill="#4285F4" d="M24 9.5c3.54 0 6.73 1.22 9.24 3.23l6.9-6.9C36.68 2.54 30.7 0 24 0 14.82 0 6.71 5.06 2.69 12.44l8.06 6.26C12.6 13.15 17.88 9.5 24 9.5z"/>
                <path fill="#34A853" d="M46.1 24.55c0-1.64-.15-3.22-.42-4.74H24v9.01h12.44c-.54 2.91-2.18 5.38-4.66 7.05l7.18 5.59C43.93 37.13 46.1 31.38 46.1 24.55z"/>
                <path fill="#FBBC05" d="M10.75 28.7c-1.13-3.36-1.13-6.97 0-10.33l-8.06-6.26C.98 16.36 0 20.06 0 24c0 3.94.98 7.64 2.69 11.06l8.06-6.36z"/>
                <path fill="#EA4335" d="M24 48c6.7 0 12.68-2.21 16.91-6.03l-7.18-5.59c-2.01 1.35-4.59 2.13-7.73 2.13-6.12 0-11.4-3.65-13.25-8.7l-8.06 6.36C6.71 42.94 14.82 48 24 48z"/>
                <path fill="none" d="M0 0h48v48H0z"/>
              </g>
            </svg>
            Sign in with Google
          </a>
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

  return (
    <ErrorBoundary>
      <div className="flex h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-950 relative">
        {/* Toast notifications */}
        {toast && (
          <Toast 
            message={toast.message} 
            type={toast.type} 
            onClose={closeToast} 
          />
        )}

        {/* Sidebar */}
        <div className="w-80 bg-white dark:bg-gray-900 shadow-xl border-r border-gray-200 dark:border-gray-800 flex flex-col">
          <div className="p-6 border-b border-gray-200 dark:border-gray-800">
            <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              LLMNet 
            </h1>
            
            {/* Mode Toggle */}
            <div className="flex mt-4 bg-gray-100 dark:bg-gray-800 rounded-lg p-1">
              <button
                onClick={() => setActiveMode('chat')}
                className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-all ${
                  activeMode === 'chat'
                    ? 'bg-white dark:bg-gray-900 text-blue-600 dark:text-blue-400 shadow-sm'
                    : 'text-gray-600 dark:text-gray-300 hover:text-gray-800 dark:hover:text-white'
                }`}
              >
                <MessageCircle className="w-4 h-4 inline mr-2" />
                Chat
              </button>
              <button
                onClick={() => setActiveMode('rag')}
                className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-all ${
                  activeMode === 'rag'
                    ? 'bg-white dark:bg-gray-900 text-purple-600 dark:text-purple-400 shadow-sm'
                    : 'text-gray-600 dark:text-gray-300 hover:text-gray-800 dark:hover:text-white'
                }`}
              >
                <FileText className="w-4 h-4 inline mr-2" />
                RAG
              </button>
            </div>
          </div>

          {/* New Chat Button */}
          <div className="p-4">
            <button
              onClick={createNewThread}
              disabled={isLoading}
              className="w-full bg-gradient-to-r from-blue-500 to-purple-600 text-white py-3 px-4 rounded-lg font-medium hover:from-blue-600 hover:to-purple-700 transition-all shadow-lg hover:shadow-xl transform hover:-translate-y-0.5 flex items-center justify-center disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
            >
              <Plus className="w-5 h-5 mr-2" />
              New {activeMode === 'rag' ? 'RAG' : 'Chat'}
            </button>
          </div>

          {/* Document Upload (RAG Mode) */}
          {activeMode === 'rag' && (
            <div className="px-4 pb-4">
              <input
                ref={fileInputRef}
                type="file"
                onChange={(e) => e.target.files[0] && handleFileUpload(e.target.files[0])}
                className="hidden"
                accept=".pdf,.txt,.doc,.docx"
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                disabled={!activeThread || isLoading}
                className="w-full bg-gradient-to-r from-green-500 to-emerald-600 text-white py-3 px-4 rounded-lg font-medium hover:from-green-600 hover:to-emerald-700 transition-all shadow-lg hover:shadow-xl transform hover:-translate-y-0.5 flex items-center justify-center disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
              >
                <Upload className="w-5 h-5 mr-2" />
                Upload Document
              </button>
              
              {uploadedDoc && (
                <div className="mt-3 p-3 bg-green-50 dark:bg-green-900 rounded-lg border border-green-200 dark:border-green-700">
                  <div className="flex items-center text-green-800 dark:text-green-200">
                    <FileText className="w-4 h-4 mr-2" />
                    <span className="text-sm font-medium truncate">{uploadedDoc.filename}</span>
                  </div>
                  <div className="text-xs text-green-600 dark:text-green-300 mt-1">
                    {uploadedDoc.chunks} chunks processed
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Settings */}
          <div className="px-4 pb-4">
            <button
              onClick={() => setShowSettings(!showSettings)}
              className="w-full bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-200 py-2 px-4 rounded-lg font-medium hover:bg-gray-200 dark:hover:bg-gray-700 transition-all flex items-center justify-center"
            >
              <Settings className="w-4 h-4 mr-2" />
              Settings
            </button>
            {showSettings && (
              <div className="mt-3 p-3 bg-gray-50 dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-700 space-y-3">
                <div>
                  <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">Model</label>
                  <select
                    value={selectedModel}
                    onChange={(e) => setSelectedModel(e.target.value)}
                    className="w-full p-2 border border-gray-300 dark:border-gray-700 rounded-md text-sm bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100"
                  >
                    <option value="llama">Llama 3.1 8B</option>
                    <option value="deepseek">DeepSeek Coder V2</option>
                  </select>
                </div>
              </div>
            )}
          </div>

          {/* Thread List */}
          <div className="flex-1 overflow-y-auto px-4">
            <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-200 mb-3">Recent Conversations</h3>
            <div className="space-y-2">
              {threads.map((thread) => (
                <button
                  key={thread.thread_id}
                  onClick={() => {
                    setActiveThread(thread.thread_id);
                    loadThreadHistory(thread.thread_id);
                  }}
                  className={`w-full text-left p-3 rounded-lg transition-all hover:bg-gray-100 dark:hover:bg-gray-800 ${
                    activeThread === thread.thread_id ? 'bg-blue-50 dark:bg-blue-900 border border-blue-200 dark:border-blue-700' : 'bg-white dark:bg-gray-900 border border-gray-100 dark:border-gray-700'
                  }`}
                >
                  <div className="text-sm font-medium text-gray-800 dark:text-gray-100 truncate">{thread.title}</div>
                  <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    {new Date(thread.created_at || thread.timestamp).toLocaleDateString()}
                  </div>
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Main Chat Area */}
        <div className="flex-1 flex flex-col">
          {/* Header */}
          <div className="bg-white dark:bg-gray-900 shadow-sm border-b border-gray-200 dark:border-gray-800 p-4">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-xl font-semibold text-gray-800 dark:text-gray-100">
                  {activeMode === 'rag' ? 'Document Q&A' : 'AI Chat'}
                </h2>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  {activeMode === 'rag' 
                    ? uploadedDoc 
                      ? `Chatting about: ${uploadedDoc.filename}`
                      : 'Upload a document to start RAG chat'
                    : 'General conversation with AI'
                  }
                </p>
              </div>
              <div className="flex items-center gap-4">
                <DarkModeToggle />
                <div className="text-sm text-gray-500 dark:text-gray-300 whitespace-nowrap">
                  Model: {selectedModel === 'llama' ? 'Llama 3.1 8B' : 'DeepSeek Coder V2'}
                </div>
                <UserCircle />
              </div>
            </div>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-6 space-y-4 bg-gray-50 dark:bg-gray-950">
            {messages.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full text-gray-500 dark:text-gray-400">
                <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center mb-4">
                  {activeMode === 'rag' ? (
                    <FileText className="w-8 h-8 text-white" />
                  ) : (
                    <MessageCircle className="w-8 h-8 text-white" />
                  )}
                </div>
                <h3 className="text-xl font-semibold mb-2">
                  {activeMode === 'rag' ? 'Start Document Q&A' : 'Start Conversation'}
                </h3>
                <p className="text-center max-w-md">
                  {activeMode === 'rag' 
                    ? 'Upload a document and ask questions about its content. I\'ll provide answers based on the document.'
                    : 'Ask me anything! I\'m here to help with questions, coding, writing, and more.'
                  }
                </p>
              </div>
            ) : (
              messages.map((message, index) => (
                <MessageBubble key={`${message.timestamp}-${index}`} message={message} />
              ))
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input Area */}
          <div className="bg-white dark:bg-gray-900 border-t border-gray-200 dark:border-gray-800 p-4">
            <div className="flex items-end space-x-4 max-w-4xl mx-auto">
              <div className="flex-1 relative">
                <textarea
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder={
                    activeMode === 'rag' && !uploadedDoc
                      ? 'Upload a document first to start asking questions...'
                      : 'Type your message... (Enter to send, Shift+Enter for new line)'
                  }
                  disabled={isLoading || (activeMode === 'rag' && !uploadedDoc)}
                  className="w-full p-4 pr-12 border border-gray-300 dark:border-gray-700 rounded-2xl resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-50 dark:disabled:bg-gray-800 disabled:text-gray-500 min-h-[56px] max-h-32 bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100"
                  rows={1}
                  style={{ height: 'auto' }}
                  onInput={(e) => {
                    e.target.style.height = 'auto';
                    e.target.style.height = Math.min(e.target.scrollHeight, 128) + 'px';
                  }}
                />
              </div>
              <button
                onClick={sendMessage}
                disabled={!inputMessage.trim() || isLoading || (activeMode === 'rag' && !uploadedDoc)}
                className="bg-gradient-to-r from-blue-500 to-purple-600 text-white p-4 rounded-2xl hover:from-blue-600 hover:to-purple-700 transition-all shadow-lg hover:shadow-xl transform hover:-translate-y-0.5 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none disabled:hover:shadow-lg"
              >
                {isLoading ? (
                  <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                ) : (
                  <Send className="w-5 h-5" />
                )}
              </button>
            </div>
          </div>
        </div>
      </div>
    </ErrorBoundary>
  );
};

export default ChatFrontend;