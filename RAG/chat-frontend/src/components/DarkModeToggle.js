/**
 * Dark Mode Toggle Component
 * Button to toggle between light and dark themes
 */

import React from 'react';
import { Moon, Sun } from 'lucide-react';

const DarkModeToggle = React.memo(({ darkMode, onToggle }) => (
  <button
    onClick={onToggle}
    className="p-2 rounded-full hover:bg-gray-200 dark:hover:bg-gray-700 transition"
    title={darkMode ? "Switch to Light Mode" : "Switch to Dark Mode"}
    aria-label={darkMode ? "Switch to Light Mode" : "Switch to Dark Mode"}
  >
    {darkMode 
      ? <Sun className="w-5 h-5 text-yellow-400" /> 
      : <Moon className="w-5 h-5 text-gray-700" />
    }
  </button>
), (prevProps, nextProps) => 
  prevProps.darkMode === nextProps.darkMode && 
  prevProps.onToggle === nextProps.onToggle
);

DarkModeToggle.displayName = 'DarkModeToggle';

export default DarkModeToggle;
