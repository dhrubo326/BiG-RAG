import api from './api';
import type { Document, DocumentMetadata } from '../types';
import { API_ENDPOINTS } from '../utils/constants';

/**
 * Get all documents with basic info (paginated)
 * Uses /documents?limit=X&offset=Y for efficient list loading
 * Does NOT fetch detailed info - use getDocumentById for that
 */
export const getDocuments = async (options?: {
  page?: number;
  limit?: number;
  dataset?: string;
}): Promise<{ documents: Document[]; total: number; page: number; limit: number }> => {
  const { page = 1, limit = 50, dataset } = options || {};
  const offset = (page - 1) * limit;

  const response = await api.get(API_ENDPOINTS.DOCUMENTS, {
    params: {
      limit,
      offset,
      dataset,
    },
  });

  const docs = response.data.documents || response.data || [];
  const total = response.data.total || docs.length;

  return {
    documents: docs,
    total,
    page,
    limit,
  };
};

/**
 * Get a single document by ID with detailed information
 * @param id - Document ID
 * @param includeEntities - Include entities list
 * @param includeRelated - Include related documents
 */
export const getDocumentById = async (
  id: string,
  includeEntities: boolean = true,
  includeRelated: boolean = true
): Promise<Document> => {
  const response = await api.get(API_ENDPOINTS.DOCUMENT_BY_ID(id), {
    params: {
      include_entities: includeEntities,
      include_related: includeRelated,
    },
  });
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

  // Add title separately (required field)
  if (metadata.title) {
    formData.append('title', metadata.title);
  }

  // Add metadata as JSON string (backend expects this format)
  const metadataObj: Record<string, any> = {};
  if (metadata.category) metadataObj.category = metadata.category;
  if (metadata.tags && metadata.tags.length > 0) metadataObj.tags = metadata.tags;
  if (metadata.author) metadataObj.author = metadata.author;
  if (metadata.source) metadataObj.source = metadata.source;
  if (metadata.url) metadataObj.url = metadata.url;

  if (Object.keys(metadataObj).length > 0) {
    formData.append('metadata', JSON.stringify(metadataObj));
  }

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
 * @param id - Document ID
 * @param hardDelete - If true, permanently removes document from all storage layers (including corpus)
 */
export const deleteDocument = async (id: string, hardDelete: boolean = false): Promise<void> => {
  await api.delete(API_ENDPOINTS.DELETE_DOCUMENT(id), {
    params: { hard_delete: hardDelete },
  });
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