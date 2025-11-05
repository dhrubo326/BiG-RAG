import api from './api';
import type { EvaluationConfig, EvaluationResult, QuestionResult } from '../types';
import { API_ENDPOINTS } from '../utils/constants';

/**
 * Run a new evaluation
 */
export const runEvaluation = async (
  config: EvaluationConfig
): Promise<{ job_id: string }> => {
  const response = await api.post(API_ENDPOINTS.EVAL_GENERATE, {
    dataset: config.dataset,
    model: config.model,
    questions: config.questions,
    top_k: config.top_k,
    enable_reranking: config.enable_reranking,
    temperature: config.temperature,
  });

  return {
    job_id: response.data.job_id || response.data.id,
  };
};

/**
 * Get evaluation results
 */
export const getEvaluationResults = async (
  jobId: string
): Promise<EvaluationResult> => {
  const response = await api.get(`${API_ENDPOINTS.EVAL_RESULTS}/${jobId}`);
  return response.data;
};

/**
 * Get all evaluation runs
 */
export const getEvaluationHistory = async (
  limit: number = 10
): Promise<EvaluationResult[]> => {
  const response = await api.get(API_ENDPOINTS.EVAL_RESULTS, {
    params: { limit },
  });

  return response.data.results || response.data || [];
};

/**
 * Evaluate specific results
 */
export const evaluateResults = async (
  csvPath: string
): Promise<{
  exact_match: number;
  f1_score: number;
  details: QuestionResult[];
}> => {
  const response = await api.post(API_ENDPOINTS.EVAL_RESULTS, {
    csv_path: csvPath,
  });

  return response.data;
};

/**
 * Compare two evaluation runs
 */
export const compareEvaluations = async (
  runId1: string,
  runId2: string
): Promise<{
  run1: EvaluationResult;
  run2: EvaluationResult;
  comparison: {
    em_diff: number;
    f1_diff: number;
    improved_questions: QuestionResult[];
    degraded_questions: QuestionResult[];
    unchanged_questions: QuestionResult[];
  };
}> => {
  const response = await api.get(`${API_ENDPOINTS.EVAL_RESULTS}/compare`, {
    params: {
      run1: runId1,
      run2: runId2,
    },
  });

  return response.data;
};

/**
 * Export evaluation results
 */
export const exportEvaluation = async (
  runId: string,
  format: 'json' | 'csv' | 'latex' = 'json'
): Promise<Blob> => {
  const response = await api.get(`${API_ENDPOINTS.EVAL_RESULTS}/${runId}/export`, {
    params: { format },
    responseType: 'blob',
  });

  return response.data;
};

/**
 * Delete evaluation run
 */
export const deleteEvaluation = async (runId: string): Promise<void> => {
  await api.delete(`${API_ENDPOINTS.EVAL_RESULTS}/${runId}`);
};

/**
 * Get evaluation metrics breakdown
 */
export const getMetricsBreakdown = async (
  runId: string
): Promise<{
  by_question_type: {
    single_passage: { em: number; f1: number; count: number };
    multi_passage: { em: number; f1: number; count: number };
    no_answer: { em: number; f1: number; count: number };
  };
  by_difficulty: {
    easy: { em: number; f1: number; count: number };
    medium: { em: number; f1: number; count: number };
    hard: { em: number; f1: number; count: number };
  };
  confusion_matrix?: number[][];
}> => {
  const response = await api.get(`${API_ENDPOINTS.EVAL_RESULTS}/${runId}/metrics`);
  return response.data;
};

/**
 * Get failed questions from evaluation
 */
export const getFailedQuestions = async (
  runId: string,
  threshold: number = 1.0
): Promise<QuestionResult[]> => {
  const response = await api.get(`${API_ENDPOINTS.EVAL_RESULTS}/${runId}/failed`, {
    params: { threshold },
  });

  return response.data.questions || response.data || [];
};

/**
 * Re-run evaluation for specific questions
 */
export const rerunEvaluation = async (
  runId: string,
  questionIds: number[]
): Promise<{ job_id: string }> => {
  const response = await api.post(`${API_ENDPOINTS.EVAL_RESULTS}/${runId}/rerun`, {
    question_ids: questionIds,
  });

  return {
    job_id: response.data.job_id || response.data.id,
  };
};

/**
 * Get evaluation job status
 */
export const getJobStatus = async (
  jobId: string
): Promise<{
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  message?: string;
  result?: any;
}> => {
  const response = await api.get(API_ENDPOINTS.JOB_BY_ID(jobId));
  return response.data;
};

/**
 * Cancel evaluation job
 */
export const cancelJob = async (jobId: string): Promise<void> => {
  await api.post(`${API_ENDPOINTS.JOB_BY_ID(jobId)}/cancel`);
};