import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

/**
 * Utility function to merge Tailwind CSS classes
 * Combines clsx and tailwind-merge for optimal class name handling
 */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}
