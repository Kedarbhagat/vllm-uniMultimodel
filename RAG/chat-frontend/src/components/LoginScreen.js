/**
 * Login Screen Component
 * Handles authentication UI
 */

import React, { useState, useCallback } from 'react';
import { AlertCircle, Loader2 } from 'lucide-react';
import { handleEmailAuth } from '../api';
import { AUTH_API_URL } from '../config';

const LoginScreen = ({ onAuthSuccess, authError, showErrorMessage }) => {
  const [showEmailLogin, setShowEmailLogin] = useState(false);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isRegister, setIsRegister] = useState(false);
  const [emailAuthLoading, setEmailAuthLoading] = useState(false);

  const handleSubmit = useCallback(async (e) => {
    e.preventDefault();
    setEmailAuthLoading(true);

    try {
      await handleEmailAuth(email, password, isRegister);
      showErrorMessage(
        `${isRegister ? 'Account created' : 'Login'} successful!`,
        'success'
      );

      setTimeout(() => {
        window.location.href = window.location.href;
      }, 500);
    } catch (error) {
      showErrorMessage(error.message);
    } finally {
      setEmailAuthLoading(false);
    }
  }, [email, password, isRegister, showErrorMessage]);

  return (
    <div className="w-screen h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center overflow-hidden">
      {/* Decorative elements */}
      <div className="absolute top-0 left-0 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl"></div>
      <div className="absolute bottom-0 right-0 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl"></div>
      
      {/* Login card */}
      <div className="relative z-10 w-full max-w-md mx-4">
        <div className="bg-slate-800/50 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-8 shadow-2xl">
          {/* Header */}
          <div className="text-center mb-8">
            <h1 className="text-4xl font-bold text-white mb-2">LLMNet</h1>
            <p className="text-slate-400 text-sm">AI-Powered Chat Assistant</p>
          </div>

          {/* Error message */}
          {authError && (
            <div className="mb-6 p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 text-sm flex items-center gap-2">
              <AlertCircle className="w-4 h-4 flex-shrink-0" />
              <span>{authError}</span>
            </div>
          )}

          {/* Login options */}
          {!showEmailLogin ? (
            <div className="space-y-4">
              {/* Email login button */}
              <button
                onClick={() => setShowEmailLogin(true)}
                className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-4 rounded-xl transition-all duration-200 shadow-lg hover:shadow-xl"
              >
                Sign in with Email
              </button>

              {/* Divider */}
              <div className="relative">
                <div className="absolute inset-0 flex items-center">
                  <div className="w-full border-t border-slate-700/50"></div>
                </div>
                <div className="relative flex justify-center text-sm">
                  <span className="px-2 bg-slate-800/50 text-slate-400">or</span>
                </div>
              </div>

              {/* Google login button — now uses AUTH_API_URL from config (port 9096) */}
              <a
                href={`${AUTH_API_URL}/login`}
                className="w-full bg-white hover:bg-slate-50 text-slate-900 font-semibold py-3 px-4 rounded-xl flex items-center justify-center gap-3 transition-all duration-200 shadow-lg hover:shadow-xl"
              >
                <svg className="w-5 h-5" viewBox="0 0 24 24">
                  <path fill="currentColor" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                  <path fill="currentColor" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                  <path fill="currentColor" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                  <path fill="currentColor" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
                </svg>
                Continue with Google
              </a>

              {/* Footer */}
              <p className="text-xs text-slate-400 text-center mt-6">
                Department of ISE, NMIT
              </p>
            </div>
          ) : (
            <form className="space-y-4" onSubmit={handleSubmit}>
              {/* Email input */}
              <div>
                <input
                  type="email"
                  className="w-full bg-slate-700/30 border border-slate-600/50 rounded-lg px-4 py-3 text-white placeholder-slate-400 focus:outline-none focus:border-blue-500/50 focus:ring-1 focus:ring-blue-500/30 transition-all duration-200"
                  placeholder="Email address"
                  value={email}
                  onChange={e => setEmail(e.target.value)}
                  required
                  autoComplete="email"
                />
              </div>

              {/* Password input */}
              <div>
                <input
                  type="password"
                  className="w-full bg-slate-700/30 border border-slate-600/50 rounded-lg px-4 py-3 text-white placeholder-slate-400 focus:outline-none focus:border-blue-500/50 focus:ring-1 focus:ring-blue-500/30 transition-all duration-200"
                  placeholder="Password"
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  required
                  autoComplete={isRegister ? "new-password" : "current-password"}
                />
              </div>

              {/* Submit button */}
              <button
                type="submit"
                className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-4 rounded-lg transition-all duration-200 shadow-lg hover:shadow-xl disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                disabled={emailAuthLoading}
              >
                {emailAuthLoading ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span>Signing in...</span>
                  </>
                ) : (
                  isRegister ? 'Create Account' : 'Sign In'
                )}
              </button>

              {/* Toggle register/login */}
              <button
                type="button"
                className="w-full text-slate-400 hover:text-slate-300 text-sm font-medium py-2 transition-colors"
                onClick={() => setIsRegister(r => !r)}
              >
                {isRegister 
                  ? 'Already have an account? Sign in' 
                  : "Don't have an account? Create one"
                }
              </button>

              {/* Back button */}
              <button
                type="button"
                className="w-full text-slate-500 hover:text-slate-400 text-xs font-medium py-2 transition-colors"
                onClick={() => setShowEmailLogin(false)}
              >
                ← Back to Google login
              </button>
            </form>
          )}
        </div>
      </div>
    </div>
  );
};

export default LoginScreen;