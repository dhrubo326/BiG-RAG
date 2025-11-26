import api from './api';
import { API_ENDPOINTS } from '../utils/constants';

export interface SubgraphInfo {
  path: string;
  description: string;
  aliases: string[];
  topics?: string[];
  enabled: boolean;
  created_at?: string;
  auto_created?: boolean;
}

export interface RegistryResponse {
  version: string;
  subgraphs: Record<string, SubgraphInfo>;
  routing_config?: {
    default_strategy?: string;
    fallback_subgraph?: string;
    max_subgraphs_per_query?: number;
  };
}

/**
 * Fetch all available datasets from unified registry
 */
export const fetchAvailableDatasets = async (): Promise<{
  datasets: Array<{ name: string; description: string; enabled: boolean }>;
  isUnifiedMode: boolean;
}> => {
  try {
    const response = await api.get<RegistryResponse>(API_ENDPOINTS.UNIFIED_REGISTRY);
    const subgraphs = response.data.subgraphs || {};

    const datasets = Object.entries(subgraphs)
      .filter(([_, config]) => config.enabled !== false)
      .map(([name, config]) => ({
        name,
        description: config.description,
        enabled: config.enabled !== false,
      }));

    return {
      datasets,
      isUnifiedMode: true,
    };
  } catch (error: any) {
    // If 503 or 404, server not in unified mode
    if (error.response?.status === 503 || error.response?.status === 404) {
      return {
        datasets: [{ name: 'demo_test', description: 'Single-mode dataset', enabled: true }],
        isUnifiedMode: false,
      };
    }
    throw error;
  }
};

/**
 * Get details for a specific dataset
 */
export const getDatasetInfo = async (datasetName: string): Promise<SubgraphInfo | null> => {
  try {
    const response = await api.get<SubgraphInfo>(
      `${API_ENDPOINTS.UNIFIED_REGISTRY}/${datasetName}`
    );
    return response.data;
  } catch {
    return null;
  }
};
