/**
 * New Chat Button Component
 * Button to create new chat session
 */

import React from 'react';
import { Plus } from 'lucide-react';

const NewChatButton = React.memo(({ darkMode, isLoading, onClick }) => (
  <div className="p-5">
    <button
      onClick={onClick}
      disabled={isLoading}
      className={`w-full py-3 px-4 rounded-xl font-semibold transition-all shadow hover:shadow-md flex items-center justify-center disabled:opacity-50 disabled:cursor-not-allowed
        ${darkMode
          ? "bg-zinc-800 text-white hover:bg-zinc-700"
          : "bg-zinc-100 text-black hover:bg-zinc-200"
        }`
      }
      aria-label="Create new chat"
    >
      <Plus className="w-5 h-5 mr-2" />
      New Chat
    </button>
  </div>
), (prevProps, nextProps) => 
  prevProps.darkMode === nextProps.darkMode && 
  prevProps.isLoading === nextProps.isLoading && 
  prevProps.onClick === nextProps.onClick
);

NewChatButton.displayName = 'NewChatButton';

export default NewChatButton;
