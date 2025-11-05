import api from './api';
import type { Document, DocumentMetadata } from '../types';
import { API_ENDPOINTS } from '../utils/constants';

/**
 * Get all documents
 */
export const getDocuments = async (): Promise<Document[]> => {
  const response = await api.get(API_ENDPOINTS.DOCUMENTS);
  return response.data.documents || response.data || [];
};

/**
 * Get a single document by ID
 */
export const getDocumentById = async (id: string): Promise<Document> => {
  const response = await api.get(API_ENDPOINTS.DOCUMENT_BY_ID(id));
  return response.data;
};

/**
 * Upload a new document
 */
export const uploadDocument = async (
  file: File,
  metadata: DocumentMetadata,
  onProgress?: (progress: number) => void
): Promise<Document> => {
  const formData = new FormData();
  formData.append('file', file);

  // Add metadata fields
  if (metadata.title) formData.append('title', metadata.title);
  if (metadata.category) formData.append('category', metadata.category);
  if (metadata.tags) formData.append('tags', JSON.stringify(metadata.tags));
  if (metadata.author) formData.append('author', metadata.author);
  if (metadata.source) formData.append('source', metadata.source);
  if (metadata.url) formData.append('url', metadata.url);

  const response = await api.post(API_ENDPOINTS.UPLOAD_DOCUMENT, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    onUploadProgress: (progressEvent) => {
      if (onProgress && progressEvent.total) {
        const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
        onProgress(progress);
      }
    },
  });

  return response.data;
};

/**
 * Upload document by text content
 */
export const uploadDocumentText = async (
  content: string,
  metadata: DocumentMetadata & { title: string }
): Promise<Document> => {
  const response = await api.post(API_ENDPOINTS.UPLOAD_DOCUMENT, {
    content,
    metadata,
  });

  return response.data;
};

/**
 * Delete a document
 */
export const deleteDocument = async (id: string): Promise<void> => {
  await api.delete(API_ENDPOINTS.DELETE_DOCUMENT(id));
};

/**
 * Delete multiple documents
 */
export const deleteMultipleDocuments = async (ids: string[]): Promise<void> => {
  // Send delete requests in parallel
  await Promise.all(ids.map((id) => deleteDocument(id)));
};

/**
 * Update document metadata
 */
export const updateDocumentMetadata = async (
  id: string,
  metadata: Partial<DocumentMetadata>
): Promise<Document> => {
  const response = await api.patch(API_ENDPOINTS.DOCUMENT_BY_ID(id), {
    metadata,
  });

  return response.data;
};

/**
 * Search documents
 */
export const searchDocuments = async (
  query: string,
  filters?: {
    type?: string[];
    source?: string[];
    tags?: string[];
  }
): Promise<Document[]> => {
  const response = await api.get(API_ENDPOINTS.DOCUMENTS, {
    params: {
      q: query,
      ...filters,
    },
  });

  return response.data.documents || response.data || [];
};

/**
 * Get document statistics
 */
export const getDocumentStats = async (): Promise<{
  total: number;
  byType: Record<string, number>;
  bySource: Record<string, number>;
  totalEntities: number;
  totalChunks: number;
}> => {
  const response = await api.get(`${API_ENDPOINTS.DOCUMENTS}/stats`);
  return response.data;
};

/**
 * Export documents
 */
export const exportDocuments = async (
  ids?: string[],
  format: 'json' | 'csv' = 'json'
): Promise<Blob> => {
  const response = await api.post(
    `${API_ENDPOINTS.DOCUMENTS}/export`,
    {
      ids,
      format,
    },
    {
      responseType: 'blob',
    }
  );

  return response.data;
};

/**
 * Import documents from file
 */
export const importDocuments = async (file: File): Promise<{
  imported: number;
  failed: number;
  errors?: string[];
}> => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await api.post(`${API_ENDPOINTS.DOCUMENTS}/import`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  return response.data;
};

/**
 * Process document with knowledge graph extraction
 */
export const processDocument = async (
  id: string
): Promise<{
  entities: number;
  relations: number;
  chunks: number;
}> => {
  const response = await api.post(`${API_ENDPOINTS.DOCUMENT_BY_ID(id)}/process`);
  return response.data;
};

/**
 * Get document chunks
 */
export const getDocumentChunks = async (id: string): Promise<{
  chunks: Array<{
    id: string;
    content: string;
    position: number;
    entities: string[];
    relations: string[];
  }>;
}> => {
  const response = await api.get(`${API_ENDPOINTS.DOCUMENT_BY_ID(id)}/chunks`);
  return response.data;
};

/**
 * Get document entities
 */
export const getDocumentEntities = async (id: string): Promise<{
  entities: Array<{
    name: string;
    type: string;
    description: string;
    occurrences: number;
    weight: number;
  }>;
}> => {
  const response = await api.get(`${API_ENDPOINTS.DOCUMENT_BY_ID(id)}/entities`);
  return response.data;
};