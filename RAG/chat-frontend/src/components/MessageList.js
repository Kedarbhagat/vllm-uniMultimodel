/**
 * Message List Component
 * Container for displaying all messages
 */

import React, { useCallback } from 'react';
import { List } from 'react-window';
import MessageBubble from './MessageBubble';
import { UI_CONFIG } from '../config';

const MessageList = React.memo(({ messages, messagesEndRef, darkMode }) => {
  const Row = useCallback(({ index, style }) => (
    <div style={style}>
      <MessageBubble message={messages[index]} darkMode={darkMode} />
    </div>
  ), [messages, darkMode]);

  // Use virtualization for large message lists
  if (messages.length > UI_CONFIG.MESSAGES_VIRTUALIZATION_THRESHOLD) {
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

  // Render all messages for smaller lists
  return (
    <div className="w-full max-w-4xl mx-auto py-8 px-4">
      {messages.map((message) => (
        <MessageBubble key={message.id} message={message} darkMode={darkMode} />
      ))}
      <div ref={messagesEndRef} />
    </div>
  );
}, (prevProps, nextProps) => prevProps.messages === nextProps.messages);

MessageList.displayName = 'MessageList';

export default MessageList;
