import { useEffect, useState } from 'react';
import { fetchAvailableDatasets } from '../services/datasets';
import { toast } from 'sonner';

export interface Dataset {
  name: string;
  description: string;
  enabled: boolean;
}

export function useDatasets() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isUnifiedMode, setIsUnifiedMode] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadDatasets = async () => {
      try {
        setIsLoading(true);
        const result = await fetchAvailableDatasets();
        setDatasets(result.datasets);
        setIsUnifiedMode(result.isUnifiedMode);
        setError(null);
      } catch (err: any) {
        console.error('Failed to fetch datasets:', err);
        setError(err.message || 'Failed to load datasets');
        toast.error('Failed to load available datasets');

        // Fallback to single mode
        setDatasets([{ name: 'demo_test', description: 'Single-mode dataset', enabled: true }]);
        setIsUnifiedMode(false);
      } finally {
        setIsLoading(false);
      }
    };

    loadDatasets();
  }, []);

  const refetch = async () => {
    try {
      setIsLoading(true);
      const result = await fetchAvailableDatasets();
      setDatasets(result.datasets);
      setIsUnifiedMode(result.isUnifiedMode);
      setError(null);
    } catch (err: any) {
      console.error('Failed to fetch datasets:', err);
      setError(err.message || 'Failed to load datasets');
    } finally {
      setIsLoading(false);
    }
  };

  return {
    datasets,
    isLoading,
    isUnifiedMode,
    error,
    refetch,
  };
}
