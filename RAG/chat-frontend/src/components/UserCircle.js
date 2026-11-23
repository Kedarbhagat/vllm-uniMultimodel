/**
 * User Circle Component
 * Displays user profile with logout option
 */

import React, { useState, useEffect } from 'react';
import { User, LogOut, Settings } from 'lucide-react';

const UserCircle = React.memo(({ userInfo, userEmail, authApiUrl, onProfileClick }) => {
  const [showLogout, setShowLogout] = useState(false);

  useEffect(() => {
    const handleClick = (e) => {
      if (!e.target.closest('.user-circle-dropdown')) {
        setShowLogout(false);
      }
    };

    if (showLogout) {
      document.addEventListener('mousedown', handleClick);
      return () => document.removeEventListener('mousedown', handleClick);
    }
  }, [showLogout]);

  return (
    <div className="relative user-circle-dropdown">
      <button
        className="w-10 h-10 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center shadow-lg border-2 border-white hover:scale-105 transition-transform focus:outline-none"
        title={userInfo?.name || userEmail}
        onClick={() => setShowLogout(v => !v)}
        aria-label="User profile"
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
        <div className="absolute right-0 mt-2 bg-white dark:bg-zinc-800 border border-gray-200 dark:border-zinc-700 shadow-lg rounded-lg z-50 min-w-[150px]">
          {(userInfo?.name || userEmail) && (
            <div className="px-4 py-2 text-xs text-gray-700 dark:text-gray-300 border-b dark:border-zinc-700">
              {userInfo?.name ? userInfo.name : userEmail}
            </div>
          )}
          <button
            onClick={() => {
              onProfileClick?.();
              setShowLogout(false);
            }}
            className="w-full flex items-center gap-2 px-4 py-2 text-gray-700 dark:text-gray-300 hover:bg-blue-50 dark:hover:bg-blue-900 hover:text-blue-600 dark:hover:text-blue-400 transition text-left"
          >
            <Settings className="w-4 h-4" />
            Profile
          </button>
          <a
            href={`${authApiUrl}/logout`}
            className="flex items-center gap-2 px-4 py-2 text-gray-700 dark:text-gray-300 hover:bg-red-50 dark:hover:bg-red-900 hover:text-red-600 dark:hover:text-red-400 rounded-b-lg transition"
          >
            <LogOut className="w-4 h-4" />
            Logout
          </a>
        </div>
      )}
    </div>
  );
}, (prevProps, nextProps) => 
  prevProps.userInfo === nextProps.userInfo && 
  prevProps.userEmail === nextProps.userEmail && 
  prevProps.authApiUrl === nextProps.authApiUrl
);

UserCircle.displayName = 'UserCircle';

export default UserCircle;
