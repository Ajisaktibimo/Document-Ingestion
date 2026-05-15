import { memo } from "react";
import { FileSearch, BookOpen, Sparkles } from "lucide-react";
import { cn } from "@/lib/utils";
import { useSessionStore } from "@/stores/useSessionStore";
import { DocumentViewer } from "./DocumentViewer";

export const VisualPanel = memo(function VisualPanel() {
  const { selectedDoc, highlight, setHighlight } = useSessionStore();

  if (!selectedDoc) {
    return (
      <div className="h-full flex flex-col items-center justify-center bg-muted/5 p-8 text-center">
        <div className="w-16 h-16 rounded-3xl bg-muted/30 flex items-center justify-center mb-6">
          <FileSearch className="w-8 h-8 text-muted-foreground/30" />
        </div>
        <h3 className="text-sm font-bold mb-2">Knowledge Viewer</h3>
        <p className="text-xs text-muted-foreground max-w-[200px] leading-relaxed">
          Select a document from the left panel to explore its parsed content and structure.
        </p>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col overflow-hidden bg-background">
      <div className="flex-shrink-0 flex items-center justify-between px-4 py-3 border-b bg-background/50 backdrop-blur-sm">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg bg-primary/10 flex items-center justify-center">
            <BookOpen className="w-4 h-4 text-primary" />
          </div>
          <span className="text-xs font-bold uppercase tracking-widest">Reader</span>
        </div>
        
        {highlight && (
          <div className="flex items-center gap-2 animate-in fade-in slide-in-from-right-2">
            <span className="flex items-center gap-1 px-2 py-1 rounded-full bg-primary/10 text-[10px] font-bold text-primary border border-primary/20">
              <Sparkles className="w-3 h-3" />
              Page {highlight.page_num}
            </span>
            <button 
              onClick={() => setHighlight(null)}
              className="text-[10px] text-muted-foreground hover:text-foreground underline underline-offset-2"
            >
              Clear
            </button>
          </div>
        )}
      </div>

      <div className="flex-1 min-h-0">
        <DocumentViewer
          doc={selectedDoc}
          highlightCitation={highlight}
          onScrolled={() => {}}
        />
      </div>
    </div>
  );
});
