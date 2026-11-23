/**
 * Error Boundary Component
 * Catches errors and displays fallback UI
 */

import React, { useState, useEffect } from 'react';
import { AlertCircle } from 'lucide-react';

const ErrorBoundary = ({ children, fallback }) => {
  const [hasError, setHasError] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const handleError = (error) => {
      setHasError(true);
      setError(error);
    };

    window.addEventListener('error', handleError);
    return () => window.removeEventListener('error', handleError);
  }, []);

  if (hasError) {
    return fallback || (
      <div className="flex items-center justify-center h-screen bg-red-50 dark:bg-red-900">
        <div className="text-center">
          <AlertCircle className="w-16 h-16 text-red-500 mx-auto mb-4" />
          <h2 className="text-xl font-bold text-red-700 dark:text-red-300 mb-2">
            Something went wrong
          </h2>
          <p className="text-red-600 dark:text-red-400">
            {error?.message || 'An unexpected error occurred'}
          </p>
          <button 
            onClick={() => window.location.reload()} 
            className="mt-4 px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
          >
            Reload Page
          </button>
        </div>
      </div>
    );
  }

  return children;
};

export default ErrorBoundary;
