import React, { useState } from 'react';
import { X, Eye, EyeOff, Check, AlertCircle } from 'lucide-react';

/**
 * UserProfile Component
 * 
 * Displays user information and allows editing password
 * 
 * @param {Object} props
 * @param {Object} props.userInfo - User information {email, name, picture, etc.}
 * @param {string} props.userEmail - User's email address
 * @param {Function} props.onClose - Callback when closing the profile screen
 * @param {Function} props.onPasswordUpdate - Callback after successful password update
 * @param {boolean} props.darkMode - Dark mode flag
 * @returns {JSX.Element} User profile screen
 */
const UserProfile = ({
  userInfo = {},
  userEmail = '',
  onClose = () => {},
  onPasswordUpdate = () => {},
  darkMode = false
}) => {
  // ============================================
  // STATE
  // ============================================
  const [isEditing, setIsEditing] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [showNewPassword, setShowNewPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  
  const [isLoading, setIsLoading] = useState(false);
  const [message, setMessage] = useState({ type: '', text: '' });

  // ============================================
  // HANDLERS
  // ============================================

  /**
   * Validate password change inputs
   */
  const validatePasswordChange = () => {
    if (!currentPassword.trim()) {
      setMessage({ type: 'error', text: 'Current password is required' });
      return false;
    }
    if (!newPassword.trim()) {
      setMessage({ type: 'error', text: 'New password is required' });
      return false;
    }
    if (newPassword.length < 6) {
      setMessage({ type: 'error', text: 'New password must be at least 6 characters' });
      return false;
    }
    if (newPassword !== confirmPassword) {
      setMessage({ type: 'error', text: 'Passwords do not match' });
      return false;
    }
    if (newPassword === currentPassword) {
      setMessage({ type: 'error', text: 'New password must be different from current password' });
      return false;
    }
    return true;
  };

  /**
   * Handle password update submission
   */
  const handlePasswordUpdate = async (e) => {
    e.preventDefault();
    
    if (!validatePasswordChange()) {
      return;
    }

    setIsLoading(true);
    setMessage({ type: '', text: '' });

    try {
      // Construct the backend URL based on current location
      // This ensures it works both locally and on external domains
      const protocol = window.location.protocol;
      let backendUrl;
      
      // Try environment variables first
      if (process.env.REACT_APP_API_BASE_URL) {
        backendUrl = process.env.REACT_APP_API_BASE_URL;
      } else if (process.env.REACT_APP_BACKEND_URL) {
        backendUrl = process.env.REACT_APP_BACKEND_URL;
      } else {
        // Fallback: Use same host as frontend but on port 9096
        const host = window.location.hostname;
        const isLocalhost = host === 'localhost' || host === '127.0.0.1';
        
        if (isLocalhost) {
          backendUrl = 'http://localhost:9096';
        } else {
          // For external domains (dev.nmit.ac.in, etc), use same domain on port 9096
          backendUrl = `${protocol}//${host}:9096`;
        }
      }
      
      const updateUrl = `${backendUrl}/api/update-password`;
      
      console.log('🔐 Password Update Request:');
      console.log('  Current Location:', window.location.href);
      console.log('  URL:', updateUrl);
      console.log('  Email:', userEmail);
      console.log('  Backend URL:', backendUrl);
      
      const response = await fetch(updateUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({
          email: userEmail,
          current_password: currentPassword,
          new_password: newPassword,
        }),
      });

      const data = await response.json();
      
      console.log('🔐 Response Status:', response.status);
      console.log('🔐 Response Data:', data);

      if (!response.ok) {
        const errorMsg = data.detail || data.message || 'Failed to update password';
        console.error('❌ Password update failed:', errorMsg);
        setMessage({ 
          type: 'error', 
          text: errorMsg
        });
        return;
      }

      setMessage({ type: 'success', text: 'Password updated successfully!' });
      
      // Clear form
      setTimeout(() => {
        setCurrentPassword('');
        setNewPassword('');
        setConfirmPassword('');
        setIsEditing(false);
        setMessage({ type: '', text: '' });
        onPasswordUpdate();
      }, 1500);
    } catch (error) {
      console.error('❌ Password update error:', error);
      setMessage({ type: 'error', text: `Error: ${error.message}` });
    } finally {
      setIsLoading(false);
    }
  };

  /**
   * Handle cancel editing
   */
  const handleCancel = () => {
    setCurrentPassword('');
    setNewPassword('');
    setConfirmPassword('');
    setMessage({ type: '', text: '' });
    setIsEditing(false);
  };

  // ============================================
  // RENDER
  // ============================================

  return (
    <div className={`fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4`}>
      <div className={`${darkMode ? 'bg-zinc-900' : 'bg-white'} rounded-lg shadow-xl max-w-md w-full max-h-[90vh] overflow-y-auto`}>
        
        {/* Header */}
        <div className={`flex items-center justify-between p-6 border-b ${darkMode ? 'border-zinc-800' : 'border-gray-200'}`}>
          <h2 className={`text-2xl font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
            User Profile
          </h2>
          <button
            onClick={onClose}
            className={`p-1 rounded-lg transition-colors ${
              darkMode
                ? 'hover:bg-zinc-800 text-zinc-400 hover:text-white'
                : 'hover:bg-gray-100 text-gray-600'
            }`}
            aria-label="Close profile"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          
          {/* User Info Display */}
          <div className={`${darkMode ? 'bg-zinc-800' : 'bg-gray-50'} p-4 rounded-lg`}>
            <div className="space-y-4">
              
              {/* Email */}
              <div>
                <label className={`block text-sm font-semibold mb-1 ${
                  darkMode ? 'text-zinc-300' : 'text-gray-700'
                }`}>
                  Email
                </label>
                <div className={`p-3 rounded-lg ${
                  darkMode 
                    ? 'bg-zinc-700 text-white' 
                    : 'bg-white border border-gray-300 text-gray-900'
                }`}>
                  {userEmail || userInfo?.email || 'N/A'}
                </div>
              </div>

              {/* Name */}
              {userInfo?.name && (
                <div>
                  <label className={`block text-sm font-semibold mb-1 ${
                    darkMode ? 'text-zinc-300' : 'text-gray-700'
                  }`}>
                    Name
                  </label>
                  <div className={`p-3 rounded-lg ${
                    darkMode 
                      ? 'bg-zinc-700 text-white' 
                      : 'bg-white border border-gray-300 text-gray-900'
                  }`}>
                    {userInfo.name}
                  </div>
                </div>
              )}

              {/* Auth Provider */}
              <div>
                <label className={`block text-sm font-semibold mb-1 ${
                  darkMode ? 'text-zinc-300' : 'text-gray-700'
                }`}>
                  Account Type
                </label>
                <div className={`p-3 rounded-lg ${
                  darkMode 
                    ? 'bg-zinc-700 text-white' 
                    : 'bg-white border border-gray-300 text-gray-900'
                }`}>
                  {userInfo?.auth_provider === 'google' ? '🔑 Google Account' : '👤 Local Account'}
                </div>
              </div>
            </div>
          </div>

          {/* Message Display */}
          {message.text && (
            <div className={`p-3 rounded-lg flex items-start gap-2 ${
              message.type === 'error'
                ? darkMode
                  ? 'bg-red-900/30 border border-red-700'
                  : 'bg-red-50 border border-red-200'
                : darkMode
                  ? 'bg-green-900/30 border border-green-700'
                  : 'bg-green-50 border border-green-200'
            }`}>
              {message.type === 'error' ? (
                <AlertCircle className={`w-5 h-5 flex-shrink-0 mt-0.5 ${
                  darkMode ? 'text-red-400' : 'text-red-600'
                }`} />
              ) : (
                <Check className={`w-5 h-5 flex-shrink-0 mt-0.5 ${
                  darkMode ? 'text-green-400' : 'text-green-600'
                }`} />
              )}
              <span className={`text-sm ${
                message.type === 'error'
                  ? darkMode ? 'text-red-300' : 'text-red-700'
                  : darkMode ? 'text-green-300' : 'text-green-700'
              }`}>
                {message.text}
              </span>
            </div>
          )}

          {/* Password Section */}
          {!isEditing ? (
            <button
              onClick={() => setIsEditing(true)}
              className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-4 rounded-lg transition-colors"
            >
              Change Password
            </button>
          ) : (
            <form onSubmit={handlePasswordUpdate} className="space-y-4">
              <h3 className={`font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                Change Password
              </h3>

              {/* Current Password */}
              <div>
                <label className={`block text-sm font-semibold mb-1 ${
                  darkMode ? 'text-zinc-300' : 'text-gray-700'
                }`}>
                  Current Password
                </label>
                <div className="relative">
                  <input
                    type={showPassword ? 'text' : 'password'}
                    value={currentPassword}
                    onChange={(e) => setCurrentPassword(e.target.value)}
                    placeholder="Enter current password"
                    className={`w-full p-3 pr-10 rounded-lg border ${
                      darkMode
                        ? 'bg-zinc-700 border-zinc-600 text-white placeholder-zinc-400'
                        : 'bg-white border-gray-300 text-gray-900'
                    } transition-colors focus:outline-none focus:border-blue-500`}
                    disabled={isLoading}
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className={`absolute right-3 top-3 ${
                      darkMode ? 'text-zinc-400' : 'text-gray-600'
                    }`}
                  >
                    {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                  </button>
                </div>
              </div>

              {/* New Password */}
              <div>
                <label className={`block text-sm font-semibold mb-1 ${
                  darkMode ? 'text-zinc-300' : 'text-gray-700'
                }`}>
                  New Password
                </label>
                <div className="relative">
                  <input
                    type={showNewPassword ? 'text' : 'password'}
                    value={newPassword}
                    onChange={(e) => setNewPassword(e.target.value)}
                    placeholder="Enter new password"
                    className={`w-full p-3 pr-10 rounded-lg border ${
                      darkMode
                        ? 'bg-zinc-700 border-zinc-600 text-white placeholder-zinc-400'
                        : 'bg-white border-gray-300 text-gray-900'
                    } transition-colors focus:outline-none focus:border-blue-500`}
                    disabled={isLoading}
                  />
                  <button
                    type="button"
                    onClick={() => setShowNewPassword(!showNewPassword)}
                    className={`absolute right-3 top-3 ${
                      darkMode ? 'text-zinc-400' : 'text-gray-600'
                    }`}
                  >
                    {showNewPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                  </button>
                </div>
              </div>

              {/* Confirm Password */}
              <div>
                <label className={`block text-sm font-semibold mb-1 ${
                  darkMode ? 'text-zinc-300' : 'text-gray-700'
                }`}>
                  Confirm New Password
                </label>
                <div className="relative">
                  <input
                    type={showConfirmPassword ? 'text' : 'password'}
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    placeholder="Confirm new password"
                    className={`w-full p-3 pr-10 rounded-lg border ${
                      darkMode
                        ? 'bg-zinc-700 border-zinc-600 text-white placeholder-zinc-400'
                        : 'bg-white border-gray-300 text-gray-900'
                    } transition-colors focus:outline-none focus:border-blue-500`}
                    disabled={isLoading}
                  />
                  <button
                    type="button"
                    onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                    className={`absolute right-3 top-3 ${
                      darkMode ? 'text-zinc-400' : 'text-gray-600'
                    }`}
                  >
                    {showConfirmPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                  </button>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex gap-3 pt-2">
                <button
                  type="button"
                  onClick={handleCancel}
                  disabled={isLoading}
                  className={`flex-1 py-3 px-4 rounded-lg font-semibold transition-colors ${
                    darkMode
                      ? 'bg-zinc-700 text-white hover:bg-zinc-600 disabled:opacity-50'
                      : 'bg-gray-200 text-gray-900 hover:bg-gray-300 disabled:opacity-50'
                  }`}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={isLoading}
                  className={`flex-1 py-3 px-4 rounded-lg font-semibold transition-colors ${
                    isLoading
                      ? 'opacity-50 cursor-not-allowed'
                      : 'bg-green-600 hover:bg-green-700 text-white'
                  }`}
                >
                  {isLoading ? 'Updating...' : 'Update Password'}
                </button>
              </div>
            </form>
          )}
        </div>
      </div>
    </div>
  );
};

export default UserProfile;
