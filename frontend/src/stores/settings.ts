import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import type { UserSettings, Dataset } from '../types';
import { DEFAULT_CONFIG } from '../utils/constants';

interface SettingsState extends UserSettings {
  // Additional state
  activeDataset: string;
  availableDatasets: Dataset[];
  isTestingConnection: boolean;
  connectionStatus: Record<string, boolean>;

  // Actions
  updateSettings: (settings: Partial<UserSettings>) => void;
  setLanguage: (language: UserSettings['language']) => void;
  setTheme: (theme: UserSettings['theme']) => void;
  setAutoSave: (autoSave: boolean) => void;
  setDefaultModel: (model: string) => void;
  setDefaultTopK: (topK: number) => void;
  setEnableReranking: (enable: boolean) => void;
  setApiKey: (provider: keyof NonNullable<UserSettings['apiKeys']>, key: string) => void;
  removeApiKey: (provider: keyof NonNullable<UserSettings['apiKeys']>) => void;
  setActiveDataset: (dataset: string) => void;
  setAvailableDatasets: (datasets: Dataset[]) => void;
  testConnection: (provider: string) => Promise<boolean>;
  resetToDefaults: () => void;
  exportSettings: () => string;
  importSettings: (settingsJson: string) => boolean;
}

const defaultSettings: UserSettings = {
  language: 'en',
  theme: 'light',
  autoSave: true,
  defaultModel: DEFAULT_CONFIG.MODEL,
  defaultTopK: DEFAULT_CONFIG.TOP_K,
  enableReranking: DEFAULT_CONFIG.ENABLE_RERANKING,
  apiKeys: {
    openai: '',
    anthropic: '',
    google: '',
    xai: '',
  },
};

const useSettingsStore = create<SettingsState>()(
  devtools(
    persist(
      (set, get) => ({
        // Initial state
        ...defaultSettings,
        activeDataset: 'demo_test',  // Server default
        availableDatasets: [],
        isTestingConnection: false,
        connectionStatus: {},

        // Actions
        updateSettings: (settings) =>
          set((state) => ({
            ...state,
            ...settings,
          })),

        setLanguage: (language) => {
          set({ language });
          // Apply language change to i18n when implemented
          // i18n.changeLanguage(language);
        },

        setTheme: (theme) => {
          set({ theme });
          // Apply theme to document root
          if (theme === 'dark') {
            document.documentElement.classList.add('dark');
          } else if (theme === 'light') {
            document.documentElement.classList.remove('dark');
          } else {
            // Auto mode - check system preference
            const prefersDark = window.matchMedia(
              '(prefers-color-scheme: dark)'
            ).matches;
            if (prefersDark) {
              document.documentElement.classList.add('dark');
            } else {
              document.documentElement.classList.remove('dark');
            }
          }
        },

        setAutoSave: (autoSave) => set({ autoSave }),

        setDefaultModel: (model) => set({ defaultModel: model }),

        setDefaultTopK: (topK) =>
          set({ defaultTopK: Math.max(1, Math.min(20, topK)) }),

        setEnableReranking: (enable) => set({ enableReranking: enable }),

        setApiKey: (provider, key) =>
          set((state) => ({
            apiKeys: {
              ...state.apiKeys,
              [provider]: key,
            },
          })),

        removeApiKey: (provider) =>
          set((state) => ({
            apiKeys: {
              ...state.apiKeys,
              [provider]: '',
            },
          })),

        setActiveDataset: (dataset) => set({ activeDataset: dataset }),

        setAvailableDatasets: (datasets) =>
          set({ availableDatasets: datasets }),

        testConnection: async (provider) => {
          set({ isTestingConnection: true });

          try {
            // Simulate API call to test connection
            // In real implementation, this would call the backend
            await new Promise((resolve) => setTimeout(resolve, 1000));

            // For demo, randomly succeed/fail
            const success = Math.random() > 0.3;

            set((state) => ({
              connectionStatus: {
                ...state.connectionStatus,
                [provider]: success,
              },
              isTestingConnection: false,
            }));

            return success;
          } catch (error) {
            set((state) => ({
              connectionStatus: {
                ...state.connectionStatus,
                [provider]: false,
              },
              isTestingConnection: false,
            }));
            return false;
          }
        },

        resetToDefaults: () => {
          const currentTheme = get().theme;
          set({
            ...defaultSettings,
            connectionStatus: {},
            isTestingConnection: false,
          });
          // Reapply theme after reset
          get().setTheme(currentTheme);
        },

        exportSettings: () => {
          const state = get();
          const exportData = {
            language: state.language,
            theme: state.theme,
            autoSave: state.autoSave,
            defaultModel: state.defaultModel,
            defaultTopK: state.defaultTopK,
            enableReranking: state.enableReranking,
            apiKeys: state.apiKeys,
            activeDataset: state.activeDataset,
          };
          return JSON.stringify(exportData, null, 2);
        },

        importSettings: (settingsJson) => {
          try {
            const settings = JSON.parse(settingsJson);
            set((state) => ({
              ...state,
              ...settings,
            }));
            // Apply theme after import
            get().setTheme(settings.theme || 'light');
            return true;
          } catch (error) {
            console.error('Failed to import settings:', error);
            return false;
          }
        },
      }),
      {
        name: 'settings-store',
        partialize: (state) => ({
          // Only persist user settings, not runtime state
          language: state.language,
          theme: state.theme,
          autoSave: state.autoSave,
          defaultModel: state.defaultModel,
          defaultTopK: state.defaultTopK,
          enableReranking: state.enableReranking,
          apiKeys: state.apiKeys,
          activeDataset: state.activeDataset,
        }),
      }
    ),
    {
      name: 'settings-store',
    }
  )
);

// Apply theme on initial load
const applyInitialTheme = () => {
  const theme = useSettingsStore.getState().theme;
  useSettingsStore.getState().setTheme(theme);
};

// Run once when the module is loaded
applyInitialTheme();

export default useSettingsStore;