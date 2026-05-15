import { memo } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import {
  MessageSquare,
  ChevronLeft,
  ChevronRight,
  Database,
} from "lucide-react";
import { useSessions } from "@/hooks/useSessions";
import { ThemeToggle } from "@/components/ui/theme-toggle";
import { cn } from "@/lib/utils";

interface SidebarProps {
  collapsed: boolean;
  onToggle: () => void;
}

export const Sidebar = memo(function Sidebar({ collapsed, onToggle }: SidebarProps) {
  const navigate = useNavigate();
  const location = useLocation();
  const { data: sessions } = useSessions();

  const activeSessionId = location.pathname.match(/\/sessions\/([a-zA-Z0-9-]+)/)?.[1];
  const isHome = location.pathname === "/";

  return (
    <aside
      className={cn(
        "flex flex-col h-full bg-card border-r border-border transition-all duration-200 flex-shrink-0",
        collapsed ? "w-14" : "w-60"
      )}
    >
      <div className="flex items-center gap-2.5 px-3 h-12 border-b border-border flex-shrink-0">
        <MessageSquare className="w-6 h-6 text-primary flex-shrink-0" />
        {!collapsed && (
          <span className="font-bold text-primary text-base truncate">DocAI</span>
        )}
      </div>

      <nav className="flex-shrink-0 px-2 pt-3 space-y-0.5">
        <button
          onClick={() => navigate("/")}
          className={cn(
            "w-full flex items-center gap-2.5 px-2.5 py-2 rounded-lg text-sm transition-colors",
            isHome && !activeSessionId
              ? "bg-primary/10 text-primary font-medium"
              : "text-muted-foreground hover:bg-muted/50 hover:text-foreground"
          )}
          title={collapsed ? "Sessions" : undefined}
        >
          <Database className="w-4 h-4 flex-shrink-0" />
          {!collapsed && <span className="truncate">Sessions</span>}
        </button>
      </nav>

      <div className="flex-1 overflow-y-auto min-h-0">
        {!collapsed && sessions && sessions.length > 0 && (
          <div className="mt-4 px-2">
            <p className="px-2.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground mb-1.5">
              Recent Chats
            </p>
            <div className="space-y-0.5">
              {sessions.map((s) => {
                const isActive = activeSessionId === s.session_id;
                return (
                  <button
                    key={s.session_id}
                    onClick={() => navigate(`/sessions/${s.session_id}`)}
                    className={cn(
                      "w-full flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-sm transition-colors",
                      isActive
                        ? "bg-primary/10 text-primary border-l-2 border-primary font-medium"
                        : "text-muted-foreground hover:bg-muted/50 hover:text-foreground"
                    )}
                  >
                    <MessageSquare className="w-3.5 h-3.5 flex-shrink-0" />
                    <span className="truncate">{s.title}</span>
                  </button>
                );
              })}
            </div>
          </div>
        )}
      </div>

      <div className="flex-shrink-0 border-t border-border px-2 py-2 flex items-center justify-between">
        <ThemeToggle />
        <button
          onClick={onToggle}
          className="w-8 h-8 flex items-center justify-center rounded-lg text-muted-foreground hover:bg-muted/50 hover:text-foreground transition-colors"
        >
          {collapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
        </button>
      </div>
    </aside>
  );
});
