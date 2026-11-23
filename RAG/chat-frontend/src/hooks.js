/**
 * Custom React Hooks
 * Reusable hooks for common functionality
 */

import { useState, useEffect, useCallback, useRef } from 'react';

/**
 * Hook for managing toast notifications
 * @returns {Object} Toast state and methods
 */
export const useToast = () => {
  const [toast, setToast] = useState(null);

  const showToast = useCallback((message, type = 'error') => {
    setToast({ message, type });
  }, []);

  const closeToast = useCallback(() => {
    setToast(null);
  }, []);

  return { toast, showToast, closeToast, setToast };
};

/**
 * Hook for managing local storage
 * @param {string} key - Storage key
 * @param {any} initialValue - Initial value
 * @returns {Array} [value, setValue]
 */
export const useLocalStorage = (key, initialValue) => {
  const [storedValue, setStoredValue] = useState(() => {
    try {
      const item = window.localStorage.getItem(key);
      return item ? JSON.parse(item) : initialValue;
    } catch (error) {
      console.error(`Error reading from localStorage [${key}]:`, error);
      return initialValue;
    }
  });

  const setValue = useCallback((value) => {
    try {
      const valueToStore = value instanceof Function ? value(storedValue) : value;
      setStoredValue(valueToStore);
      window.localStorage.setItem(key, JSON.stringify(valueToStore));
    } catch (error) {
      console.error(`Error writing to localStorage [${key}]:`, error);
    }
  }, [key, storedValue]);

  return [storedValue, setValue];
};

/**
 * Hook for scroll-to-bottom functionality
 * @param {number} dependency - Dependency to trigger scroll
 * @returns {Object} { messagesEndRef, scrollToBottom }
 */
export const useScrollToBottom = (dependency) => {
  const messagesEndRef = useRef(null);

  const scrollToBottom = useCallback(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, []);

  useEffect(() => {
    const timer = setTimeout(() => {
      scrollToBottom();
    }, 100);

    return () => clearTimeout(timer);
  }, [dependency, scrollToBottom]);

  return { messagesEndRef, scrollToBottom };
};

/**
 * Hook for outside click detection
 * @param {React.MutableRefObject} ref - Ref to element
 * @param {Function} callback - Function to call on outside click
 * @param {boolean} isActive - Whether to listen
 */
export const useOutsideClick = (ref, callback, isActive = true) => {
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (ref.current && !ref.current.contains(event.target)) {
        callback();
      }
    };

    if (isActive) {
      document.addEventListener('mousedown', handleClickOutside);
      return () => document.removeEventListener('mousedown', handleClickOutside);
    }
  }, [ref, callback, isActive]);
};

/**
 * Hook for dark mode management
 * @returns {Object} { darkMode, setDarkMode }
 */
export const useDarkMode = () => {
  const [darkMode, setDarkModeState] = useLocalStorage('darkMode', true);

  const setDarkMode = useCallback((value) => {
    const isDark = value instanceof Function ? value(darkMode) : value;
    setDarkModeState(isDark);
    
    if (isDark) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [darkMode, setDarkModeState]);

  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [darkMode]);

  return { darkMode, setDarkMode };
};

/**
 * Hook for debounced value
 * @param {any} value - Value to debounce
 * @param {number} delay - Debounce delay in ms
 * @returns {any} Debounced value
 */
export const useDebounce = (value, delay = 500) => {
  const [debouncedValue, setDebouncedValue] = useState(value);

  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    return () => clearTimeout(handler);
  }, [value, delay]);

  return debouncedValue;
};

/**
 * Hook for mounting detection
 * @returns {boolean} Is component mounted
 */
export const useIsMounted = () => {
  const [isMounted, setIsMounted] = useState(false);

  useEffect(() => {
    setIsMounted(true);
  }, []);

  return isMounted;
};

/**
 * Hook for previous value
 * @param {any} value - Current value
 * @returns {any} Previous value
 */
export const usePrevious = (value) => {
  const ref = useRef();

  useEffect(() => {
    ref.current = value;
  }, [value]);

  return ref.current;
};

/**
 * Hook for async operation
 * @param {Function} asyncFunction - Async function to execute
 * @param {Array} dependencies - Dependencies
 * @returns {Object} { data, loading, error }
 */
export const useAsync = (asyncFunction, dependencies = []) => {
  const [state, setState] = useState({
    data: null,
    loading: true,
    error: null
  });

  useEffect(() => {
    let isMounted = true;

    const executeAsync = async () => {
      try {
        const response = await asyncFunction();
        if (isMounted) {
          setState({ data: response, loading: false, error: null });
        }
      } catch (error) {
        if (isMounted) {
          setState({ data: null, loading: false, error });
        }
      }
    };

    executeAsync();

    return () => {
      isMounted = false;
    };
  }, dependencies);

  return state;
};
