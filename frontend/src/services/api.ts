import axios, { type AxiosInstance, type AxiosError, type InternalAxiosRequestConfig } from 'axios';
import { toast } from 'sonner';

// Get API URL from environment or use default
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001';

// Create axios instance with base configuration
const api: AxiosInstance = axios.create({
  baseURL: API_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor - add auth token, logging
api.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    // Log request in development
    if (import.meta.env.DEV) {
      console.log(`→ ${config.method?.toUpperCase()} ${config.url}`);
    }

    // Add auth token if available
    const token = localStorage.getItem('auth_token');
    if (token && config.headers) {
      config.headers.Authorization = `Bearer ${token}`;
    }

    return config;
  },
  (error: AxiosError) => {
    console.error('Request error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor - error handling, toasts
api.interceptors.response.use(
  (response) => {
    // Log response in development
    if (import.meta.env.DEV) {
      console.log(`← ${response.status} ${response.config.url}`);
    }
    return response;
  },
  (error: AxiosError<{ detail?: string; message?: string }>) => {
    // Extract error message
    const message =
      error.response?.data?.detail ||
      error.response?.data?.message ||
      error.message ||
      'An unexpected error occurred';

    // Show error toast (except for 401 which might be expected)
    if (error.response?.status !== 401) {
      toast.error(`API Error: ${message}`);
    }

    // Handle specific error codes
    switch (error.response?.status) {
      case 401:
        // Unauthorized - clear auth and redirect to login if needed
        localStorage.removeItem('auth_token');
        // window.location.href = '/login'; // Uncomment if you have login page
        break;
      case 404:
        console.error('Resource not found:', error.config?.url);
        break;
      case 500:
        console.error('Server error:', message);
        break;
      default:
        console.error('API error:', error);
    }

    return Promise.reject(error);
  }
);

export default api;

// Health check function
export const checkAPIHealth = async (): Promise<boolean> => {
  try {
    const response = await api.get('/');
    return response.status === 200;
  } catch {
    return false;
  }
};