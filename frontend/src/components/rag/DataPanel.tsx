import { useState, useMemo, useCallback, memo } from "react";
import { toast } from "sonner";
import {
  FileText,
  Plus,
  Trash2,
  Check,
  Search,
  Upload,
  Loader2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { ConfirmDialog } from "@/components/ui/confirm-dialog";
import { cn } from "@/lib/utils";
import type { Document, Session } from "@/types";

interface DataPanelProps {
  session: Session | undefined;
  documents: Document[] | undefined;
  docsLoading: boolean;
  selectedDocId: string | null;
  onSelectDoc: (doc: Document) => void;
  onToggleInSession: (docId: string) => void;
  onUpload: (file: File) => void;
  isUploading: boolean;
  onDelete: (id: string) => void;
}

export const DataPanel = memo(function DataPanel({
  session,
  documents,
  docsLoading,
  selectedDocId,
  onSelectDoc,
  onToggleInSession,
  onUpload,
  isUploading,
  onDelete,
}: DataPanelProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const [deleteDocConfirm, setDeleteDocConfirm] = useState<string | null>(null);

  const filteredDocs = useMemo(() => {
    if (!documents) return [];
    if (!searchQuery.trim()) return documents;
    const q = searchQuery.toLowerCase();
    return documents.filter((d) =>
      d.original_filename.toLowerCase().includes(q)
    );
  }, [documents, searchQuery]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) onUpload(file);
  };

  return (
    <div className="h-full flex flex-col border-r bg-card/30 overflow-hidden">
      <div className="p-4 border-b space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-bold flex items-center gap-2">
            <FileText className="w-4 h-4 text-primary" />
            Documents
          </h2>
          <div className="relative">
            <input
              type="file"
              id="file-upload"
              className="hidden"
              onChange={handleFileChange}
              accept=".pdf,.docx,.txt"
            />
            <Button
              size="sm"
              variant="outline"
              className="h-8 rounded-full"
              disabled={isUploading}
              onClick={() => document.getElementById("file-upload")?.click()}
            >
              {isUploading ? (
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
              ) : (
                <Plus className="w-3.5 h-3.5 mr-1" />
              )}
              Upload
            </Button>
          </div>
        </div>

        <div className="relative">
          <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search documents..."
            className="pl-9 h-9 bg-background/50"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-3 space-y-2">
        {docsLoading ? (
          <div className="flex flex-col items-center justify-center py-10 text-muted-foreground">
            <Loader2 className="w-6 h-6 animate-spin mb-2" />
            <span className="text-xs">Loading library...</span>
          </div>
        ) : filteredDocs.length === 0 ? (
          <div className="text-center py-10 px-4">
            <p className="text-xs text-muted-foreground">
              {searchQuery ? "No matches found" : "No documents in library yet."}
            </p>
          </div>
        ) : (
          filteredDocs.map((doc) => {
            const isSelected = selectedDocId === doc.id;
            const isInSession = session?.doc_ids?.includes(doc.id);
            const isProcessing = doc.status === "processing" || doc.status === "parsing";
            const stageLabel = doc.ingestion_stage?.replace(/_/g, " ");

            return (
              <div
                key={doc.id}
                onClick={() => onSelectDoc(doc)}
                className={cn(
                  "group relative p-3 rounded-xl border transition-all cursor-pointer",
                  isSelected
                    ? "bg-primary/10 border-primary/30 ring-1 ring-primary/20"
                    : "bg-background/50 border-border/50 hover:border-primary/20 hover:bg-background"
                )}
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="min-w-0 flex-1">
                    <p className="text-sm font-medium truncate leading-tight">
                      {doc.original_filename}
                    </p>
                    <p className="text-[10px] text-muted-foreground mt-1 uppercase tracking-wider">
                      {doc.status}{stageLabel ? ` / ${stageLabel}` : ""} / {new Date(doc.created_at).toLocaleDateString()}
                    </p>
                    {isProcessing && doc.ingestion_heartbeat_at && (
                      <p className="text-[10px] text-muted-foreground/80 mt-1">
                        Updated {new Date(doc.ingestion_heartbeat_at).toLocaleTimeString()}
                      </p>
                    )}
                    {doc.status === "failed" && doc.error_message && (
                      <p className="text-[10px] text-destructive mt-1 line-clamp-2">
                        {doc.error_message}
                      </p>
                    )}
                  </div>
                  
                  <div className="flex items-center gap-1">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onToggleInSession(doc.id);
                      }}
                      className={cn(
                        "w-7 h-7 flex items-center justify-center rounded-lg transition-all",
                        isInSession
                          ? "bg-primary text-primary-foreground shadow-sm"
                          : "bg-muted text-muted-foreground hover:bg-primary/20 hover:text-primary"
                      )}
                      title={isInSession ? "Remove from session context" : "Add to session context"}
                    >
                      <Check className={cn("w-3.5 h-3.5", !isInSession && "opacity-0 group-hover:opacity-100")} />
                    </button>
                    
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setDeleteDocConfirm(doc.id);
                      }}
                      className="w-7 h-7 flex items-center justify-center rounded-lg text-muted-foreground hover:bg-destructive/10 hover:text-destructive opacity-0 group-hover:opacity-100 transition-all"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>

                {isProcessing && (
                  <div className="absolute bottom-0 left-0 h-0.5 bg-primary animate-shimmer w-full" />
                )}
              </div>
            );
          })
        )}
      </div>

      <ConfirmDialog
        open={deleteDocConfirm !== null}
        onConfirm={() => {
          if (deleteDocConfirm) onDelete(deleteDocConfirm);
          setDeleteDocConfirm(null);
        }}
        onCancel={() => setDeleteDocConfirm(null)}
        title="Delete Document"
        message="This will permanently delete the document and all its indexed content. This cannot be undone."
        variant="danger"
      />
    </div>
  );
});
