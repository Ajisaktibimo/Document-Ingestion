import { memo } from "react";
import { useLocation } from "react-router-dom";
import { ChevronRight, Database, MessageSquare } from "lucide-react";
import { cn } from "@/lib/utils";

interface TopBarProps {
  actions?: React.ReactNode;
  className?: string;
}

export const TopBar = memo(function TopBar({ actions, className }: TopBarProps) {
  const location = useLocation();

  const segments: { label: string; active: boolean }[] = [
    { label: "DocAI", active: false },
  ];

  if (location.pathname === "/") {
    segments.push({ label: "Sessions", active: true });
  } else if (location.pathname.startsWith("/sessions/")) {
    segments.push({ label: "Chat", active: true });
  }

  return (
    <div
      className={cn(
        "h-12 flex items-center justify-between px-4 border-b border-border flex-shrink-0 bg-background",
        className
      )}
    >
      <div className="flex items-center gap-1.5 text-sm min-w-0">
        {segments.map((seg, i) => (
          <div key={i} className="flex items-center gap-1.5 min-w-0">
            {i > 0 && <ChevronRight className="w-3.5 h-3.5 text-muted-foreground flex-shrink-0" />}
            <span
              className={cn(
                "truncate",
                seg.active ? "font-medium text-foreground" : "text-muted-foreground"
              )}
            >
              {seg.label}
            </span>
          </div>
        ))}
      </div>

      <div className="flex items-center gap-2 flex-shrink-0">
        {actions}
      </div>
    </div>
  );
});
