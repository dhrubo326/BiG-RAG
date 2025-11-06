/**
 * Dashboard Page
 *
 * Main landing page showing:
 * - System health and stats
 * - Knowledge graph overview
 * - Recent queries
 * - Quick actions
 */

import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router';
import {
  Database,
  GitBranch,
  FileText,
  Activity,
  MessageSquare,
  Upload,
  Search,
  BarChart3,
  RefreshCw
} from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Badge } from '../components/ui/badge';
import { checkAPIHealth } from '../services/api';
import api from '../services/api';
import { toast } from 'sonner';

interface GraphStats {
  success: boolean;
  total_datasets: number;
  global_stats: {
    total_documents: number;
    total_entities: number;
    total_edges: number;
    total_chunks: number;
  };
  datasets: Array<{
    dataset: string;
    total_documents: number;
    indexed_documents: number;
    pending_documents: number;
    failed_documents: number;
    total_chunks: number;
    total_entities: number;
    total_edges: number;
    total_tokens: number;
  }>;
}

export function Dashboard() {
  const navigate = useNavigate();
  const [stats, setStats] = useState<GraphStats | null>(null);
  const [apiStatus, setApiStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  const [isLoading, setIsLoading] = useState(true);

  const checkHealth = async () => {
    const isHealthy = await checkAPIHealth();
    setApiStatus(isHealthy ? 'online' : 'offline');
  };

  const loadStats = async () => {
    setIsLoading(true);
    try {
      // Fetch graph statistics
      const response = await api.get<GraphStats>('/graph/stats');
      setStats(response.data);
    } catch (error) {
      console.error('Failed to load stats:', error);
      toast.error('Failed to load statistics');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    checkHealth();
    loadStats();

    // Refresh stats every 30 seconds
    const interval = setInterval(() => {
      checkHealth();
      loadStats();
    }, 30000);

    return () => clearInterval(interval);
  }, []);

  const handleRefresh = () => {
    checkHealth();
    loadStats();
    toast.success('Dashboard refreshed');
  };

  return (
    <div className="container mx-auto p-6 max-w-7xl">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-4xl font-bold mb-2">BiG-RAG Dashboard</h1>
          <p className="text-gray-600 dark:text-gray-400">
            Bipartite Graph Retrieval-Augmented Generation System
          </p>
        </div>
        <Button onClick={handleRefresh} variant="outline" size="sm">
          <RefreshCw className="w-4 h-4 mr-2" />
          Refresh
        </Button>
      </div>

      {/* Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-500 dark:text-gray-400">
                  API Status
                </p>
                <div className="flex items-center gap-2 mt-2">
                  {apiStatus === 'checking' ? (
                    <Badge variant="secondary">Checking...</Badge>
                  ) : apiStatus === 'online' ? (
                    <Badge variant="success">Online</Badge>
                  ) : (
                    <Badge variant="destructive">Offline</Badge>
                  )}
                </div>
              </div>
              <Activity className={`w-8 h-8 ${
                apiStatus === 'online' ? 'text-green-600' : 'text-red-600'
              }`} />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-500 dark:text-gray-400">
                  Total Documents
                </p>
                <p className="text-3xl font-bold mt-2">
                  {isLoading ? '...' : stats?.global_stats.total_documents || 0}
                </p>
              </div>
              <FileText className="w-8 h-8 text-blue-600" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-500 dark:text-gray-400">
                  Total Entities
                </p>
                <p className="text-3xl font-bold mt-2">
                  {isLoading ? '...' : stats?.global_stats.total_entities || 0}
                </p>
              </div>
              <Database className="w-8 h-8 text-purple-600" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-500 dark:text-gray-400">
                  Total Relations
                </p>
                <p className="text-3xl font-bold mt-2">
                  {isLoading ? '...' : stats?.global_stats.total_edges || 0}
                </p>
              </div>
              <GitBranch className="w-8 h-8 text-red-600" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Quick Actions */}
      <Card className="mb-8">
        <CardHeader>
          <CardTitle>Quick Actions</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <Button
              onClick={() => navigate('/chat')}
              variant="outline"
              className="h-auto py-4 flex-col gap-2"
            >
              <MessageSquare className="w-6 h-6" />
              <span>Ask a Question</span>
            </Button>

            <Button
              onClick={() => navigate('/documents')}
              variant="outline"
              className="h-auto py-4 flex-col gap-2"
            >
              <Upload className="w-6 h-6" />
              <span>Upload Document</span>
            </Button>

            <Button
              onClick={() => navigate('/graph')}
              variant="outline"
              className="h-auto py-4 flex-col gap-2"
            >
              <Search className="w-6 h-6" />
              <span>Explore Graph</span>
            </Button>

            <Button
              onClick={() => navigate('/evaluation')}
              variant="outline"
              className="h-auto py-4 flex-col gap-2"
            >
              <BarChart3 className="w-6 h-6" />
              <span>View Analytics</span>
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Datasets Overview */}
      {stats && stats.datasets && stats.datasets.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Datasets Overview</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {stats.datasets.map((dataset) => (
                <div
                  key={dataset.dataset}
                  className="border border-gray-200 dark:border-gray-700 rounded-lg p-4"
                >
                  <div className="flex items-center justify-between mb-3">
                    <div>
                      <h3 className="font-semibold text-lg">{dataset.dataset}</h3>
                      <p className="text-sm text-gray-500 dark:text-gray-400">
                        {dataset.indexed_documents} / {dataset.total_documents} documents indexed
                      </p>
                    </div>
                    <div className="flex gap-2">
                      {dataset.pending_documents > 0 && (
                        <Badge variant="warning">
                          {dataset.pending_documents} pending
                        </Badge>
                      )}
                      {dataset.failed_documents > 0 && (
                        <Badge variant="destructive">
                          {dataset.failed_documents} failed
                        </Badge>
                      )}
                      {dataset.pending_documents === 0 && dataset.failed_documents === 0 && (
                        <Badge variant="success">Ready</Badge>
                      )}
                    </div>
                  </div>

                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                    <div>
                      <p className="text-gray-500 dark:text-gray-400">Chunks</p>
                      <p className="font-semibold">{dataset.total_chunks.toLocaleString()}</p>
                    </div>
                    <div>
                      <p className="text-gray-500 dark:text-gray-400">Entities</p>
                      <p className="font-semibold">{dataset.total_entities.toLocaleString()}</p>
                    </div>
                    <div>
                      <p className="text-gray-500 dark:text-gray-400">Relations</p>
                      <p className="font-semibold">{dataset.total_edges.toLocaleString()}</p>
                    </div>
                    <div>
                      <p className="text-gray-500 dark:text-gray-400">Tokens</p>
                      <p className="font-semibold">{dataset.total_tokens.toLocaleString()}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Getting Started Guide */}
      {(!stats || stats.global_stats.total_documents === 0) && !isLoading && (
        <Card>
          <CardHeader>
            <CardTitle>Getting Started</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <p className="text-gray-600 dark:text-gray-400">
                Welcome to BiG-RAG! Your knowledge graph is empty. Here's how to get started:
              </p>
              <ol className="list-decimal list-inside space-y-2 text-gray-700 dark:text-gray-300">
                <li>Upload documents using the "Upload Document" button above</li>
                <li>Wait for the system to process and extract entities</li>
                <li>Explore the graph visualization to see connections</li>
                <li>Start asking questions in the Chat interface</li>
              </ol>
              <Button onClick={() => navigate('/documents')}>
                Upload Your First Document
              </Button>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
