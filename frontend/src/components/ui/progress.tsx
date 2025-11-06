import * as React from "react"
import { cn } from "../../utils/cn"

export interface ProgressProps extends React.HTMLAttributes<HTMLDivElement> {
  value?: number;
  max?: number;
  showValue?: boolean;
}

const Progress = React.forwardRef<HTMLDivElement, ProgressProps>(
  ({ className, value = 0, max = 100, showValue = false, ...props }, ref) => {
    const percentage = Math.min(Math.max((value / max) * 100, 0), 100);

    return (
      <div ref={ref} className="w-full" {...props}>
        <div
          className={cn(
            "relative h-2 w-full overflow-hidden rounded-full bg-gray-200 dark:bg-gray-700",
            className
          )}
        >
          <div
            className="h-full bg-blue-600 transition-all duration-300 ease-in-out"
            style={{ width: `${percentage}%` }}
          />
        </div>
        {showValue && (
          <div className="mt-1 text-xs text-gray-500 dark:text-gray-400 text-right">
            {Math.round(percentage)}%
          </div>
        )}
      </div>
    )
  }
)
Progress.displayName = "Progress"

export { Progress }
