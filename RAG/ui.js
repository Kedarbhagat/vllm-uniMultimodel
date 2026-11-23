import React, { useState, useEffect, useRef } from 'react';
import { Send, Upload, FileText, User, Bot, Settings, MessageCircle, Plus, Trash2, Search } from 'lucide-react';

const API_BASE_URL = 'http://localhost:9075'; // Adjust to your FastAPI server URL

const ChatFrontend = () => {
  const [activeMode, setActiveMode] = useState('chat'); // 'chat' or 'rag'
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [threads, setThreads] = useState([]);
  const [activeThread, setActiveThread] = useState(null);
  const [userEmail, setUserEmail] = useState('user@example.com');
  const [selectedModel, setSelectedModel] = useState('llama');
  const [uploadedDoc, setUploadedDoc] = useState(null);
  const [showSettings, setShowSettings] = useState(false);
  
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    createInitialThread();
    loadUserThreads();
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const createInitialThread = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/chat/create_thread`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          email: userEmail,
          title: 'New Chat'
        })
      });
      const data = await response.json();
      setActiveThread(data.thread_id);
    } catch (error) {
      console.error('Error creating thread:', error);
    }
  };

  const loadUserThreads = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/chat/threads?email=${userEmail}`);
      const data = await response.json();
      setThreads(data);
    } catch (error) {
      console.error('Error loading threads:', error);
    }
  };

  const loadThreadHistory = async (threadId) => {
    try {
      const endpoint = activeMode === 'rag' ? 
        `${API_BASE_URL}/chat/rag_history?thread_id=${threadId}` :
        `${API_BASE_URL}/chat/history?thread_id=${threadId}`;
      
      const response = await fetch(endpoint);
      const data = await response.json();
      
      if (activeMode === 'rag' && data.messages) {
        setMessages(data.messages.map(msg => ({
          role: msg.role,
          content: msg.content,
          timestamp: msg.timestamp
        })));
      } else {
        setMessages(data.map(msg => ({
          role: msg.role,
          content: msg.content,
          timestamp: msg.created_at
        })));
      }
    } catch (error) {
      console.error('Error loading history:', error);
    }
  };

  const createNewThread = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/chat/create_thread`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          email: userEmail,
          title: `New ${activeMode === 'rag' ? 'RAG' : 'Chat'}`
        })
      });
      const data = await response.json();
      setActiveThread(data.thread_id);
      setMessages([]);
      setUploadedDoc(null);
      loadUserThreads();
    } catch (error) {
      console.error('Error creating new thread:', error);
    }
  };

  const handleFileUpload = async (file) => {
    if (!activeThread || activeMode !== 'rag') return;

    const formData = new FormData();
    formData.append('thread_id', activeThread);
    formData.append('file', file);

    try {
      setIsLoading(true);
      const response = await fetch(`${API_BASE_URL}/chat/upload_doc`, {
        method: 'POST',
        body: formData
      });
      const data = await response.json();
      
      if (data.document_id) {
        setUploadedDoc({
          id: data.document_id,
          filename: data.filename,
          chunks: data.chunks
        });
        
        // Add system message about upload
        const uploadMsg = {
          role: 'system',
          content: `📄 Document "${data.filename}" uploaded successfully. ${data.chunks} chunks processed. Ready for Q&A!`,
          timestamp: new Date().toISOString()
        };
        setMessages(prev => [...prev, uploadMsg]);
      }
    } catch (error) {
      console.error('Error uploading file:', error);
      const errorMsg = {
        role: 'system',
        content: `❌ Failed to upload document: ${error.message}`,
        timestamp: new Date().toISOString()
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  const sendMessage = async () => {
    if (!inputMessage.trim() || !activeThread || isLoading) return;

    const userMessage = {
      role: 'human',
      content: inputMessage,
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMessage]);
    const currentMessage = inputMessage;
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

      if (!response.body) throw new Error('No response body');

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

        const chunk = decoder.decode(value);
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
              }
            } catch (e) {
              // Ignore parsing errors for streaming
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
      const errorMsg = {
        role: 'system',
        content: `❌ Error: ${error.message}`,
        timestamp: new Date().toISOString()
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const MessageBubble = ({ message }) => {
    const isUser = message.role === 'human';
    const isSystem = message.role === 'system';
    
    return (
      <div className={`flex mb-6 ${isUser ? 'justify-end' : 'justify-start'}`}>
        {!isUser && !isSystem && (
          <div className="flex-shrink-0 w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center mr-3">
            <Bot className="w-4 h-4 text-white" />
          </div>
        )}
        
        <div className={`max-w-xs lg:max-w-md px-4 py-3 rounded-2xl relative ${
          isUser 
            ? 'bg-gradient-to-br from-blue-500 to-blue-600 text-white' 
            : isSystem 
            ? 'bg-gradient-to-br from-orange-100 to-orange-200 text-orange-800 border border-orange-300'
            : 'bg-white text-gray-800 shadow-lg border border-gray-100'
        }`}>
          <div className="text-sm leading-relaxed whitespace-pre-wrap">
            {message.content}
          </div>
          {message.streaming && (
            <div className="inline-flex items-center mt-2">
              <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse"></div>
              <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse ml-1 delay-75"></div>
              <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse ml-1 delay-150"></div>
            </div>
          )}
          <div className={`text-xs mt-2 ${isUser ? 'text-blue-100' : isSystem ? 'text-orange-600' : 'text-gray-500'}`}>
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
  };

  return (
    <div className="flex h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      {/* Sidebar */}
      <div className="w-80 bg-white shadow-xl border-r border-gray-200 flex flex-col">
        <div className="p-6 border-b border-gray-200">
          <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
            AI Chat Assistant
          </h1>
          
          {/* Mode Toggle */}
          <div className="flex mt-4 bg-gray-100 rounded-lg p-1">
            <button
              onClick={() => setActiveMode('chat')}
              className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-all ${
                activeMode === 'chat'
                  ? 'bg-white text-blue-600 shadow-sm'
                  : 'text-gray-600 hover:text-gray-800'
              }`}
            >
              <MessageCircle className="w-4 h-4 inline mr-2" />
              Chat
            </button>
            <button
              onClick={() => setActiveMode('rag')}
              className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-all ${
                activeMode === 'rag'
                  ? 'bg-white text-purple-600 shadow-sm'
                  : 'text-gray-600 hover:text-gray-800'
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
            className="w-full bg-gradient-to-r from-blue-500 to-purple-600 text-white py-3 px-4 rounded-lg font-medium hover:from-blue-600 hover:to-purple-700 transition-all shadow-lg hover:shadow-xl transform hover:-translate-y-0.5 flex items-center justify-center"
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
              className="w-full bg-gradient-to-r from-green-500 to-emerald-600 text-white py-3 px-4 rounded-lg font-medium hover:from-green-600 hover:to-emerald-700 transition-all shadow-lg hover:shadow-xl transform hover:-translate-y-0.5 flex items-center justify-center disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Upload className="w-5 h-5 mr-2" />
              Upload Document
            </button>
            
            {uploadedDoc && (
              <div className="mt-3 p-3 bg-green-50 rounded-lg border border-green-200">
                <div className="flex items-center text-green-800">
                  <FileText className="w-4 h-4 mr-2" />
                  <span className="text-sm font-medium truncate">{uploadedDoc.filename}</span>
                </div>
                <div className="text-xs text-green-600 mt-1">
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
            className="w-full bg-gray-100 text-gray-700 py-2 px-4 rounded-lg font-medium hover:bg-gray-200 transition-all flex items-center justify-center"
          >
            <Settings className="w-4 h-4 mr-2" />
            Settings
          </button>
          
          {showSettings && (
            <div className="mt-3 p-3 bg-gray-50 rounded-lg border border-gray-200 space-y-3">
              <div>
                <label className="block text-xs font-medium text-gray-700 mb-1">Model</label>
                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className="w-full p-2 border border-gray-300 rounded-md text-sm"
                >
                  <option value="llama">Llama 3.1 8B</option>
                  <option value="deepseek">DeepSeek Coder V2</option>
                </select>
              </div>
              <div>
                <label className="block text-xs font-medium text-gray-700 mb-1">Email</label>
                <input
                  type="email"
                  value={userEmail}
                  onChange={(e) => setUserEmail(e.target.value)}
                  className="w-full p-2 border border-gray-300 rounded-md text-sm"
                />
              </div>
            </div>
          )}
        </div>

        {/* Thread List */}
        <div className="flex-1 overflow-y-auto px-4">
          <h3 className="text-sm font-semibold text-gray-700 mb-3">Recent Conversations</h3>
          <div className="space-y-2">
            {threads.map((thread) => (
              <button
                key={thread.thread_id}
                onClick={() => {
                  setActiveThread(thread.thread_id);
                  loadThreadHistory(thread.thread_id);
                }}
                className={`w-full text-left p-3 rounded-lg transition-all hover:bg-gray-100 ${
                  activeThread === thread.thread_id ? 'bg-blue-50 border border-blue-200' : 'bg-white border border-gray-100'
                }`}
              >
                <div className="text-sm font-medium text-gray-800 truncate">{thread.title}</div>
                <div className="text-xs text-gray-500 mt-1">Click to load history</div>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="bg-white shadow-sm border-b border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-xl font-semibold text-gray-800">
                {activeMode === 'rag' ? 'Document Q&A' : 'AI Chat'}
              </h2>
              <p className="text-sm text-gray-600">
                {activeMode === 'rag' 
                  ? uploadedDoc 
                    ? `Chatting about: ${uploadedDoc.filename}`
                    : 'Upload a document to start RAG chat'
                  : 'General conversation with AI'
                }
              </p>
            </div>
            <div className="text-sm text-gray-500">
              Model: {selectedModel === 'llama' ? 'Llama 3.1 8B' : 'DeepSeek Coder V2'}
            </div>
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-gray-500">
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
              <MessageBubble key={index} message={message} />
            ))
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="bg-white border-t border-gray-200 p-4">
          <div className="flex items-end space-x-4 max-w-4xl mx-auto">
            <div className="flex-1 relative">
              <textarea
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                  }
                }}
                placeholder={
                  activeMode === 'rag' && !uploadedDoc
                    ? 'Upload a document first to start asking questions...'
                    : 'Type your message... (Enter to send, Shift+Enter for new line)'
                }
                disabled={isLoading || (activeMode === 'rag' && !uploadedDoc)}
                className="w-full p-4 pr-12 border border-gray-300 rounded-2xl resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-50 disabled:text-gray-500 min-h-[56px] max-h-32"
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
  );
};

export default ChatFrontend;