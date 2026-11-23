/**
 * Message Bubble Component
 * Displays individual messages in the chat
 */

import React, { useMemo } from 'react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import remarkGfm from 'remark-gfm';

// Markdown components
const MARKDOWN_COMPONENTS = {
  code({ inline, className, children, ...props }) {
    const language = /language-(\w+)/.exec(className || "")?.[1];
    
    if (inline) {
      return (
        <code className="bg-gray-100 dark:bg-gray-800 rounded px-1.5 py-0.5 font-mono text-sm" {...props}>
          {children}
        </code>
      );
    }
    
    return (
      <SyntaxHighlighter
        style={oneDark}
        language={language || 'text'}
        PreTag="div"
        customStyle={{
          borderRadius: "0.5rem",
          padding: "1rem",
          margin: "1rem 0",
        }}
      >
        {String(children).replace(/\n$/, "")}
      </SyntaxHighlighter>
    );
  },
};

const MessageBubble = React.memo(({ message, darkMode }) => {
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
    const hasReasoningOnly = message.reasoning && message.reasoning.trim() !== '' && (!message.content || message.content.trim() === '');
    const hasContent = message.content && message.content.trim() !== '';

    return (
      <div className="w-full mb-10 px-4">
        <div className="max-w-4xl mx-auto">
          {/* Reasoning Section */}
          {message.reasoning && message.reasoning.trim() !== '' && (
            <div className={`mb-4 p-4 rounded-lg border flex flex-col ${
              message.streaming && hasReasoningOnly
                ? 'bg-blue-50 dark:bg-blue-950 border-blue-200 dark:border-blue-800'
                : 'bg-zinc-100 dark:bg-zinc-800/50 border-zinc-200 dark:border-zinc-700'
            }`} style={{ maxHeight: '400px', overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
              <div className="flex items-start gap-2 flex-1 overflow-hidden flex-col">
                <div className="flex items-center gap-2 flex-shrink-0">
                  <span className="text-lg">💭</span>
                  <div className="font-semibold text-zinc-700 dark:text-zinc-300 text-sm">
                    {message.streaming && hasReasoningOnly ? 'Thinking...' : 'Reasoning'}
                  </div>
                </div>
                <div className="text-zinc-700 dark:text-zinc-300 whitespace-pre-wrap font-mono text-xs leading-relaxed overflow-y-auto flex-1 w-full pr-2" style={{ scrollBehavior: 'smooth' }}>
                  {message.reasoning}
                  {message.streaming && <span className="animate-pulse">▌</span>}
                </div>
              </div>
            </div>
          )}

          {/* Response Content */}
          {hasContent && (
            <div className="prose prose-base dark:prose-invert max-w-none text-zinc-800 dark:text-zinc-200 [&>p]:leading-relaxed [&>ul]:leading-relaxed [&>ol]:leading-relaxed [&>table]:leading-relaxed">
              {renderContent}
              {message.streaming && hasContent && (
                <div className="flex items-center mt-2 text-blue-600 dark:text-blue-400 text-xs">
                  <span className="inline-flex items-center gap-1">
                    <span className="w-1.5 h-1.5 bg-blue-600 dark:bg-blue-400 rounded-full animate-bounce"></span>
                    <span className="w-1.5 h-1.5 bg-blue-600 dark:bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></span>
                    <span className="w-1.5 h-1.5 bg-blue-600 dark:bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></span>
                  </span>
                </div>
              )}
            </div>
          )}

          {/* Streaming Indicator */}
          {message.streaming && hasReasoningOnly && (
            <div className="flex items-center mt-4 text-blue-600 dark:text-blue-400 text-xs">
              <span className="inline-flex items-center gap-1">
                <span className="w-1.5 h-1.5 bg-blue-600 dark:bg-blue-400 rounded-full animate-bounce"></span>
                <span className="w-1.5 h-1.5 bg-blue-600 dark:bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></span>
                <span className="w-1.5 h-1.5 bg-blue-600 dark:bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></span>
              </span>
            </div>
          )}
        </div>
      </div>
    );
  }

  if (isUser) {
    return (
      <div className="w-full mb-8 px-4">
        <div className="max-w-4xl mx-auto flex justify-end">
          <div className="w-96 bg-gray-200 dark:bg-gray-700 rounded-lg p-4 text-black dark:text-white overflow-hidden">
            <div style={{ userSelect: "text", wordWrap: "break-word", overflowWrap: "break-word" }}>
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  ...MARKDOWN_COMPONENTS,
                  p: ({ children }) => <p className="break-words whitespace-normal">{children}</p>,
                  strong: ({ children }) => <strong className="font-bold">{children}</strong>,
                  em: ({ children }) => <em className="italic">{children}</em>,
                }}
              >
                {message.content}
              </ReactMarkdown>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // System message
  return (
    <div className="w-full mb-8 px-4">
      <div className="max-w-4xl mx-auto flex justify-center">
        <div className="text-center text-xs text-zinc-500 dark:text-zinc-400 italic">
          {renderContent}
        </div>
      </div>
    </div>  
  );
}, (prevProps, nextProps) =>
  prevProps.message.id === nextProps.message.id &&
  prevProps.message.content === nextProps.message.content &&
  prevProps.message.reasoning === nextProps.message.reasoning &&
  prevProps.message.streaming === nextProps.message.streaming &&
  prevProps.darkMode === nextProps.darkMode
);

MessageBubble.displayName = 'MessageBubble';

export default MessageBubble;
