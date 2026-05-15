import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Brain,
  Lightbulb,
  Search,
  Database,
  PenLine,
  CheckCircle2,
  AlertCircle,
  Loader2,
  ChevronDown,
  type LucideIcon,
} from "lucide-react";
import { cn } from "@/lib/utils";
import type { AgentStep, AgentStepType } from "@/types";

interface StepConfig {
  icon: LucideIcon;
  label: string;
}

const STEP_CONFIG: Record<AgentStepType, StepConfig> = {
  analyzing: { icon: Brain, label: "Analyzing" },
  understood: { icon: Lightbulb, label: "Understood" },
  retrieving: { icon: Search, label: "Searching" },
  sources_found: { icon: Database, label: "Sources found" },
  generating: { icon: PenLine, label: "Generating" },
  done: { icon: CheckCircle2, label: "Done" },
  error: { icon: AlertCircle, label: "Error" },
};

function formatMs(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  return `${Math.floor(ms / 60000)}m ${Math.floor((ms % 60000) / 1000)}s`;
}

function LiveTimer({ startTimestamp }: { startTimestamp: number }) {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    const iv = setInterval(() => setElapsed(Date.now() - startTimestamp), 100);
    return () => clearInterval(iv);
  }, [startTimestamp]);
  return (
    <span className="text-[11px] font-mono tabular-nums text-primary/80">
      {formatMs(elapsed)}
    </span>
  );
}

function ThinkingLogSection({ text }: { text: string }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <div className="mt-1.5">
      <button
        onClick={() => setExpanded((p) => !p)}
        className="flex items-center gap-1 text-[11px] text-muted-foreground/70 hover:text-muted-foreground transition-colors"
      >
        <Brain className="w-2.5 h-2.5" />
        <span>{expanded ? "Hide" : "Show"} thinking log</span>
        <ChevronDown className={cn("w-2.5 h-2.5 transition-transform", expanded && "rotate-180")} />
      </button>
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden"
          >
            <div className={cn(
              "mt-1 ml-1 text-[11px] leading-relaxed text-muted-foreground/80 italic",
              "max-h-[200px] overflow-y-auto scrollbar-none",
              "border-l border-border/40 pl-2",
              "whitespace-pre-wrap break-words",
            )}>
              {text}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function StepNode({ step, isLast, isLive }: { step: AgentStep; isLast: boolean; isLive: boolean }) {
  const config = STEP_CONFIG[step.step];
  const Icon = config.icon;
  const isActive = step.status === "active";
  const isError = step.status === "error";
  const isCompleted = step.status === "completed";

  return (
    <motion.div
      initial={{ opacity: 0, x: -6 }}
      animate={{ opacity: 1, x: 0 }}
      className="flex gap-2 relative"
    >
      {!isLast && (
        <div className={cn("absolute left-[9px] top-[20px] w-px bottom-0", isActive ? "bg-primary/20" : "bg-border/50")} />
      )}
      <div className="relative flex-shrink-0 z-10">
        {isActive ? (
          <div className="w-[18px] h-[18px] rounded-full bg-primary/15 flex items-center justify-center ring-1 ring-primary/30">
            <Loader2 className="w-2.5 h-2.5 animate-spin text-primary" />
          </div>
        ) : isError ? (
          <div className="w-[18px] h-[18px] rounded-full bg-destructive/15 flex items-center justify-center">
            <AlertCircle className="w-2.5 h-2.5 text-destructive" />
          </div>
        ) : step.step === "done" ? (
          <div className="w-[18px] h-[18px] rounded-full bg-emerald-500/15 flex items-center justify-center">
            <CheckCircle2 className="w-2.5 h-2.5 text-emerald-500" />
          </div>
        ) : (
          <div className="w-[18px] h-[18px] rounded-full bg-muted flex items-center justify-center">
            <Icon className="w-2.5 h-2.5 text-muted-foreground/80" />
          </div>
        )}
      </div>
      <div className="flex-1 min-w-0 pb-2.5">
        <div className="flex items-center gap-1.5 min-h-[18px]">
          <span className={cn(
            "text-xs leading-tight",
            isActive && "text-foreground font-medium",
            isCompleted && step.step !== "done" && "text-muted-foreground",
            step.step === "done" && "text-emerald-500 font-medium",
            isError && "text-destructive font-medium",
          )}>
            {step.detail}
          </span>
          <span className="ml-auto flex-shrink-0">
            {isActive && isLive ? (
              <LiveTimer startTimestamp={step.timestamp} />
            ) : step.durationMs != null && step.durationMs > 0 ? (
              <span className="text-[11px] font-mono tabular-nums text-muted-foreground/70">
                {formatMs(step.durationMs)}
              </span>
            ) : null}
          </span>
        </div>
        {step.step === "analyzing" && step.thinkingText && !isActive && (
          <ThinkingLogSection text={step.thinkingText} />
        )}
      </div>
    </motion.div>
  );
}

function buildSummary(steps: AgentStep[]): string {
  const sourcesStep = steps.find((s) => s.step === "sources_found");
  const doneStep = steps.find((s) => s.step === "done");
  const parts: string[] = [];
  if (sourcesStep) {
    parts.push(`${sourcesStep.sourceCount || 0} source${(sourcesStep.sourceCount || 0) > 1 ? "s" : ""}`);
  }
  if (doneStep?.durationMs) {
    parts.push(formatMs(doneStep.durationMs));
  }
  const activeStep = steps.find((s) => s.status === "active");
  if (parts.length === 0) {
    if (activeStep) {
      const cfg = STEP_CONFIG[activeStep.step];
      return cfg ? `${cfg.label}...` : "Processing...";
    }
    return "Processed";
  }
  if (sourcesStep) {
    const suffix = parts[1] ? ` in ${parts[1]}` : activeStep ? " — generating..." : "";
    return `Found ${parts[0]}${suffix}`;
  }
  return `Completed in ${parts[0]}`;
}

export function ThinkingTimeline({
  steps,
  mode,
  className,
  autoCollapse = false,
}: {
  steps: AgentStep[];
  mode: "live" | "embedded";
  className?: string;
  autoCollapse?: boolean;
}) {
  const [expanded, setExpanded] = useState(mode === "live");
  const hasAutoCollapsedRef = useRef(false);

  useEffect(() => {
    if (autoCollapse && !hasAutoCollapsedRef.current) {
      hasAutoCollapsedRef.current = true;
      setExpanded(false);
    }
  }, [autoCollapse]);

  if (steps.length === 0) return null;

  const isStillActive = steps.some((s) => s.status === "active");
  if (!expanded) {
    return (
      <div className={cn("rounded-md border border-border/60 bg-background overflow-hidden", className)}>
        <button
          onClick={() => setExpanded(true)}
          className="w-full flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-medium text-primary/80 hover:text-primary transition-colors"
        >
          {isStillActive ? (
            <Loader2 className="w-3 h-3 animate-spin text-primary/80 flex-shrink-0" />
          ) : (
            <CheckCircle2 className="w-3 h-3 text-emerald-500/80 flex-shrink-0" />
          )}
          <span className="flex-1 text-left">{buildSummary(steps)}</span>
          <ChevronDown className="w-3 h-3 flex-shrink-0" />
        </button>
      </div>
    );
  }

  const isEmbedded = mode === "embedded" || autoCollapse;

  return (
    <div className={cn("relative", isEmbedded && "rounded-md border border-border/60 bg-background overflow-hidden", className)}>
      {isEmbedded && (
        <button
          onClick={() => setExpanded(false)}
          className="w-full flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-medium text-primary/80 hover:text-primary transition-colors border-b border-border/40"
        >
          <CheckCircle2 className="w-3 h-3 text-emerald-500/80 flex-shrink-0" />
          <span className="flex-1 text-left">{buildSummary(steps)}</span>
          <ChevronDown className="w-3 h-3 flex-shrink-0 rotate-180" />
        </button>
      )}
      <div className={cn(isEmbedded && "px-2.5 py-2")}>
        <AnimatePresence mode="popLayout">
          {steps.map((step, i) => (
            <StepNode key={step.id} step={step} isLast={i === steps.length - 1} isLive={mode === "live"} />
          ))}
        </AnimatePresence>
      </div>
    </div>
  );
}
