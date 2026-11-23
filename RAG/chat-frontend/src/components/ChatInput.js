/**
 * Chat Input Component
 * Input area for sending messages and uploading files
 */

import React, { useState, useRef, useCallback } from 'react';
import { Send, Loader2, Paperclip, Square } from 'lucide-react';

const ChatInput = React.memo(({ onSendMessage, isLoading, onFileUpload, isUploading, activeThread, onStopResponse }) => {
  const [inputValue, setInputValue] = useState('');
  const fileInputRef = useRef(null);

  const handleInputChange = useCallback((e) => {
    setInputValue(e.target.value);
    e.target.style.height = 'auto';
    e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px';
  }, []);

  const handleSend = useCallback(() => {
    if (inputValue.trim() && !isLoading && !isUploading) {
      onSendMessage(inputValue.trim());
      setInputValue('');
    }
  }, [inputValue, isLoading, isUploading, onSendMessage]);

  const handleKeyDown = useCallback((e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }, [handleSend]);

  return (
    <div className="p-2">
      <div className="max-w-3xl mx-auto">
        <div className="relative flex items-end bg-white dark:bg-zinc-900 rounded-2xl shadow-md border border-zinc-200 dark:border-zinc-800 overflow-hidden">
          {/* File Upload Button */}
          <div className="p-2">
            <input
              ref={fileInputRef}
              type="file"
              onChange={(e) => e.target.files[0] && onFileUpload(e.target.files[0])}
              className="hidden"
              accept=".pdf,.txt,.doc,.docx,.xlsx,.xls,.csv"
              aria-label="Upload document"
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={!activeThread || isUploading}
              className="p-2 rounded-full hover:bg-gray-100 dark:hover:bg-zinc-800 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              title="Upload document (PDF, TXT, DOC, DOCX)"
              aria-label="Upload file"
            >
              {isUploading ? (
                <Loader2 className="w-5 h-5 animate-spin text-gray-500" />
              ) : (
                <Paperclip className="w-5 h-5 text-gray-500 dark:text-gray-400" />
              )}
            </button>
          </div>

          {/* Input Field */}
          <textarea
            value={inputValue}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            placeholder={isUploading ? "Uploading document... please wait" : "Ask anything..."}
            disabled={isLoading || isUploading}
            className="bg-transparent resize-none focus:outline-none text-base text-zinc-900 dark:text-zinc-100 placeholder-zinc-400 dark:placeholder-zinc-500 py-3 min-h-[48px] max-w-full w-full disabled:opacity-50 disabled:cursor-not-allowed"
            rows={1}
            style={{ maxHeight: '120px' }}
            aria-label="Message input"
          />

          {/* Send/Stop Button */}
          <div className="p-2">
            {isLoading ? (
              <button
                onClick={onStopResponse}
                className="bg-zinc-700 dark:bg-zinc-600 text-white p-2 rounded-full shadow-lg hover:bg-zinc-800 dark:hover:bg-zinc-500 transition-all duration-200 flex items-center justify-center w-10 h-10 border border-zinc-600 dark:border-zinc-500"
                aria-label="Stop response"
                title="Stop response (Esc)"
              >
                <Square className="w-4 h-4 fill-current" />
              </button>
            ) : (
              <button
                onClick={handleSend}
                disabled={!inputValue.trim() || isUploading}
                className="bg-gradient-to-r from-blue-500 to-purple-600 text-white p-2 rounded-full shadow-lg hover:from-blue-600 hover:to-purple-700 transition-all duration-200 flex items-center justify-center disabled:opacity-50 disabled:cursor-not-allowed w-10 h-10"
                aria-label="Send message"
                title={isUploading ? "Please wait for document upload to complete" : "Send message"}
              >
                <Send className="w-4 h-4" />
              </button>
            )}
          </div>
        </div>
        <p className="text-xs text-gray-500 dark:text-gray-400 mt-2 text-center">
          AI can make mistakes. Please verify important information. {isLoading && "• Press Esc to stop"}
        </p>

      </div>
    </div>
  );
});

ChatInput.displayName = 'ChatInput';

export default ChatInput;
