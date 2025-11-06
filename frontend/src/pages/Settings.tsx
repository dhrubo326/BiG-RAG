/**
 * Settings Page
 *
 * Application configuration:
 * - API endpoint configuration
 * - Retrieval mode settings (hybrid, local, global, naive)
 * - Reranking toggle
 * - Theme preferences (light/dark)
 * - Language selection
 * - top_k and other query parameters
 */

import { useState } from 'react';
import { Save, RotateCcw, Check, AlertCircle, Key, Database, Palette, Zap } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { Label } from '../components/ui/label';
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '../components/ui/select';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '../components/ui/tabs';
import { Slider } from '../components/ui/slider';
import { Badge } from '../components/ui/badge';
import useSettingsStore from '../stores/settings';
import { toast } from 'sonner';
import { AVAILABLE_MODELS, QUERY_MODES, THEMES, LANGUAGES } from '../utils/constants';
import api from '../services/api';

export function Settings() {
  const {
    apiEndpoint,
    openaiApiKey,
    dataset,
    defaultModel,
    defaultTemperature,
    defaultTopK,
    defaultQueryMode,
    defaultEnableReranking,
    theme,
    language,
    setApiEndpoint,
    setOpenaiApiKey,
    setDataset,
    setDefaultModel,
    setDefaultTemperature,
    setDefaultTopK,
    setDefaultQueryMode,
    setDefaultEnableReranking,
    setTheme,
    setLanguage,
    resetToDefaults,
  } = useSettingsStore();

  const [localApiKey, setLocalApiKey] = useState(openaiApiKey || '');
  const [showApiKey, setShowApiKey] = useState(false);
  const [testingConnection, setTestingConnection] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<'idle' | 'success' | 'error'>('idle');

  const handleSave = () => {
    // Save settings to localStorage via Zustand middleware
    if (localApiKey && localApiKey !== openaiApiKey) {
      setOpenaiApiKey(localApiKey);
    }
    toast.success('Settings saved successfully');
  };

  const handleReset = () => {
    if (window.confirm('Are you sure you want to reset all settings to defaults?')) {
      resetToDefaults();
      setLocalApiKey('');
      toast.success('Settings reset to defaults');
    }
  };

  const testConnection = async () => {
    setTestingConnection(true);
    setConnectionStatus('idle');

    try {
      const response = await api.get('/');
      if (response.status === 200) {
        setConnectionStatus('success');
        toast.success('Connection successful!');
      }
    } catch (error) {
      setConnectionStatus('error');
      toast.error('Connection failed. Please check your API endpoint.');
    } finally {
      setTestingConnection(false);
    }
  };

  return (
    <div className="container mx-auto p-6 max-w-5xl">
      {/* Header */}
      <div className="flex justify-between items-center mb-8">
        <div>
          <h1 className="text-4xl font-bold mb-2">Settings</h1>
          <p className="text-gray-600 dark:text-gray-400">
            Configure your BiG-RAG application preferences
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={handleReset}>
            <RotateCcw className="w-4 h-4 mr-2" />
            Reset
          </Button>
          <Button onClick={handleSave}>
            <Save className="w-4 h-4 mr-2" />
            Save Settings
          </Button>
        </div>
      </div>

      {/* Settings Tabs */}
      <Tabs defaultValue="general" className="space-y-6">
        <TabsList>
          <TabsTrigger value="general">
            <Zap className="w-4 h-4 mr-2" />
            General
          </TabsTrigger>
          <TabsTrigger value="api">
            <Key className="w-4 h-4 mr-2" />
            API Keys
          </TabsTrigger>
          <TabsTrigger value="retrieval">
            <Database className="w-4 h-4 mr-2" />
            Retrieval
          </TabsTrigger>
          <TabsTrigger value="appearance">
            <Palette className="w-4 h-4 mr-2" />
            Appearance
          </TabsTrigger>
        </TabsList>

        {/* General Tab */}
        <TabsContent value="general">
          <Card>
            <CardHeader>
              <CardTitle>General Settings</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* API Endpoint */}
              <div>
                <Label htmlFor="api-endpoint">API Endpoint</Label>
                <div className="flex gap-2 mt-2">
                  <Input
                    id="api-endpoint"
                    value={apiEndpoint}
                    onChange={(e) => setApiEndpoint(e.target.value)}
                    placeholder="http://localhost:8001"
                  />
                  <Button
                    variant="outline"
                    onClick={testConnection}
                    disabled={testingConnection}
                  >
                    {testingConnection ? 'Testing...' : 'Test'}
                  </Button>
                </div>
                {connectionStatus === 'success' && (
                  <p className="text-sm text-green-600 dark:text-green-400 mt-1 flex items-center gap-1">
                    <Check className="w-4 h-4" />
                    Connection successful
                  </p>
                )}
                {connectionStatus === 'error' && (
                  <p className="text-sm text-red-600 dark:text-red-400 mt-1 flex items-center gap-1">
                    <AlertCircle className="w-4 h-4" />
                    Connection failed
                  </p>
                )}
              </div>

              {/* Dataset */}
              <div>
                <Label htmlFor="dataset">Default Dataset</Label>
                <Select value={dataset} onValueChange={setDataset}>
                  <SelectTrigger id="dataset" className="mt-2">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="SingleTopic">SingleTopic</SelectItem>
                    <SelectItem value="demo_test">Demo Test</SelectItem>
                    <SelectItem value="2WikiMultiHopQA">2WikiMultiHopQA</SelectItem>
                    <SelectItem value="HotpotQA">HotpotQA</SelectItem>
                    <SelectItem value="Musique">Musique</SelectItem>
                  </SelectContent>
                </Select>
                <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                  The dataset to use for queries and graph visualization
                </p>
              </div>

              {/* Default Model */}
              <div>
                <Label htmlFor="model">Default LLM Model</Label>
                <Select value={defaultModel} onValueChange={setDefaultModel}>
                  <SelectTrigger id="model" className="mt-2">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {AVAILABLE_MODELS.map((model) => (
                      <SelectItem key={model.value} value={model.value}>
                        {model.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* API Keys Tab */}
        <TabsContent value="api">
          <Card>
            <CardHeader>
              <CardTitle>API Keys</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
                <p className="text-sm text-blue-800 dark:text-blue-200">
                  API keys are stored locally in your browser and never sent to any server except the configured endpoints.
                </p>
              </div>

              {/* OpenAI API Key */}
              <div>
                <Label htmlFor="openai-key">OpenAI API Key</Label>
                <div className="flex gap-2 mt-2">
                  <Input
                    id="openai-key"
                    type={showApiKey ? 'text' : 'password'}
                    value={localApiKey}
                    onChange={(e) => setLocalApiKey(e.target.value)}
                    placeholder="sk-..."
                  />
                  <Button
                    variant="outline"
                    onClick={() => setShowApiKey(!showApiKey)}
                  >
                    {showApiKey ? 'Hide' : 'Show'}
                  </Button>
                </div>
                <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                  Required for chat completions if using OpenAI models
                </p>
              </div>

              {openaiApiKey && (
                <div className="flex items-center gap-2 text-green-600 dark:text-green-400">
                  <Check className="w-4 h-4" />
                  <span className="text-sm">API key configured</span>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Retrieval Tab */}
        <TabsContent value="retrieval">
          <Card>
            <CardHeader>
              <CardTitle>Retrieval Settings</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Query Mode */}
              <div>
                <Label htmlFor="query-mode">Default Query Mode</Label>
                <Select value={defaultQueryMode} onValueChange={setDefaultQueryMode}>
                  <SelectTrigger id="query-mode" className="mt-2">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {QUERY_MODES.map((mode) => (
                      <SelectItem key={mode.value} value={mode.value}>
                        {mode.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                  {defaultQueryMode === 'hybrid' && 'Combines entity, relation, and chunk-based retrieval (recommended)'}
                  {defaultQueryMode === 'local' && 'Entity-based retrieval only'}
                  {defaultQueryMode === 'global' && 'Relation-based retrieval only'}
                  {defaultQueryMode === 'naive' && 'Direct chunk retrieval without graph structure'}
                </p>
              </div>

              {/* Top K */}
              <div>
                <Slider
                  label="Top-K Results"
                  min={1}
                  max={20}
                  step={1}
                  value={defaultTopK}
                  onChange={(e) => setDefaultTopK(Number(e.target.value))}
                  showValue
                />
                <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                  Number of context items to retrieve per query
                </p>
              </div>

              {/* Temperature */}
              <div>
                <Slider
                  label="Temperature"
                  min={0}
                  max={1}
                  step={0.1}
                  value={defaultTemperature}
                  onChange={(e) => setDefaultTemperature(Number(e.target.value))}
                  showValue
                />
                <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                  Controls randomness in LLM responses (0 = deterministic, 1 = creative)
                </p>
              </div>

              {/* Enable Reranking */}
              <div className="flex items-center justify-between p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
                <div className="flex-1">
                  <Label htmlFor="reranking" className="text-base">
                    Semantic Reranking
                  </Label>
                  <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                    Use cross-encoder to rerank chunk candidates for better relevance
                  </p>
                </div>
                <input
                  id="reranking"
                  type="checkbox"
                  checked={defaultEnableReranking}
                  onChange={(e) => setDefaultEnableReranking(e.target.checked)}
                  className="w-5 h-5 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                />
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Appearance Tab */}
        <TabsContent value="appearance">
          <Card>
            <CardHeader>
              <CardTitle>Appearance Settings</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Theme */}
              <div>
                <Label htmlFor="theme">Theme</Label>
                <Select value={theme} onValueChange={setTheme}>
                  <SelectTrigger id="theme" className="mt-2">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {THEMES.map((t) => (
                      <SelectItem key={t.value} value={t.value}>
                        {t.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                  Choose your preferred color scheme
                </p>
              </div>

              {/* Language */}
              <div>
                <Label htmlFor="language">Language</Label>
                <Select value={language} onValueChange={setLanguage}>
                  <SelectTrigger id="language" className="mt-2">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {LANGUAGES.map((lang) => (
                      <SelectItem key={lang.value} value={lang.value}>
                        {lang.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                  Select your preferred language (i18n coming soon)
                </p>
              </div>

              {/* Preview */}
              <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                <p className="text-sm font-medium mb-3">Preview</p>
                <div className="flex gap-2">
                  <Badge>Default</Badge>
                  <Badge variant="success">Success</Badge>
                  <Badge variant="warning">Warning</Badge>
                  <Badge variant="destructive">Error</Badge>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
