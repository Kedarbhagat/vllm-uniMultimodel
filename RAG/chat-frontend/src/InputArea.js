import React, { useRef } from 'react';
import { Send, Loader2, Paperclip } from 'lucide-react';

const InputArea = ({
  inputMessage,
  handleInputChange,
  handleKeyDown,
  sendMessage,
  isLoading,
  fileInputRef,
  handleFileUpload,
  activeThread,
  isUploading
}) => (
  <div className="p-6">
    <div className="max-w-3xl mx-auto">
      <div className="relative flex items-end bg-white dark:bg-zinc-900 rounded-2xl shadow-md border border-zinc-200 dark:border-zinc-800 overflow-hidden">
        {/* File Upload Button */}
        <div className="p-2">
          <input
            ref={fileInputRef}
            type="file"
            onChange={(e) => e.target.files[0] && handleFileUpload(e.target.files[0])}
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

        {/* Text Input */}
        <textarea
          value={inputMessage}
          onChange={handleInputChange}
          onKeyDown={handleKeyDown}
          placeholder="Ask anything..."
          disabled={isLoading}
          className="bg-transparent resize-none focus:outline-none text-base text-zinc-900 dark:text-zinc-100 placeholder-zinc-400 dark:placeholder-zinc-500 py-3 min-h-[48px] max-w-full w-full disabled:opacity-50 disabled:cursor-not-allowed"
          rows={1}
          style={{ maxHeight: '120px' }}
        />

        {/* Send Button */}
        <div className="p-2">
          <button
            onClick={sendMessage}
            disabled={!inputMessage.trim() || isLoading}
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

export default React.memo(InputArea);