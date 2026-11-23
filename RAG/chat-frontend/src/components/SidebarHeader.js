/**
 * Sidebar Header Component
 * Header section of the sidebar with logo and close button
 */

import React from 'react';
import { X } from 'lucide-react';

const SidebarHeader = React.memo(({ darkMode, onClose }) => (
  <div className="p-7 border-b border-zinc-200 dark:border-zinc-800 flex items-center justify-between">
    <h1 className={`text-3xl font-extrabold tracking-tight ${
      darkMode ? "text-white" : "text-black"
    }`}>
      LLMNet
    </h1>
    <button
      onClick={onClose}
      className="lg:hidden p-2 hover:bg-zinc-100 dark:hover:bg-zinc-800 rounded-lg transition-colors"
      aria-label="Close sidebar"
    >
      <X className="w-5 h-5 text-zinc-600 dark:text-zinc-400" />
    </button>
  </div>
), (prevProps, nextProps) => 
  prevProps.darkMode === nextProps.darkMode && 
  prevProps.onClose === nextProps.onClose
);

SidebarHeader.displayName = 'SidebarHeader';

export default SidebarHeader;
