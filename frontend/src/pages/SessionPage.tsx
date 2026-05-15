import { useMemo, useCallback, useEffect } from "react";
import { useParams } from "react-router-dom";
import { toast } from "sonner";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { DataPanel } from "@/components/rag/DataPanel";
import { ChatPanel } from "@/components/rag/ChatPanel";
import { VisualPanel } from "@/components/rag/VisualPanel";
import { useSessionStore } from "@/stores/useSessionStore";
import { useSession, usePatchSession } from "@/hooks/useSessions";
import { api } from "@/lib/api";
import type { Document } from "@/types";

export function SessionPage() {
  const { sessionId } = useParams<{ sessionId: string }>();
  const queryClient = useQueryClient();

  const { data: session } = useSession(sessionId);
  const patchSession = usePatchSession();

  const { selectedDoc, selectDoc, reset: resetStore } = useSessionStore();

  useEffect(() => {
    resetStore();
  }, [sessionId, resetStore]);

  // Fetch all documents available in the system
  const { data: globalDocuments, isLoading: docsLoading } = useQuery({
    queryKey: ["documents"],
    queryFn: () => api.get<Document[]>("/documents"),
    refetchInterval: (query) => {
      const docs = query.state.data;
      if (docs?.some((d) => d.status === "processing" || d.status === "parsing")) return 3000;
      return false;
    },
  });

  const uploadDoc = useMutation({
    mutationFn: (file: File) => api.uploadFile<Document>("/upload", file),
    onSuccess: (newDoc) => {
      queryClient.invalidateQueries({ queryKey: ["documents"] });
      toast.success("Document uploaded");
      // Automatically add to session if possible
      if (session && sessionId) {
        const currentIds = session.doc_ids || [];
        patchSession.mutate({
          session_id: sessionId,
          body: { doc_ids: [...currentIds, newDoc.id] }
        });
      }
    },
    onError: () => toast.error("Upload failed"),
  });

  const deleteDoc = useMutation({
    mutationFn: (id: string) => api.delete(`/documents/${id}`),
    onSuccess: (_, id) => {
      queryClient.invalidateQueries({ queryKey: ["documents"] });
      if (selectedDoc?.id === id) selectDoc(null);
      toast.success("Document deleted");
    },
    onError: () => toast.error("Delete failed"),
  });

  const handleSelectDoc = useCallback(
    (doc: Document) => {
      if (selectedDoc?.id === doc.id) {
        selectDoc(null);
      } else {
        selectDoc(doc);
      }
    },
    [selectedDoc, selectDoc]
  );

  const handleToggleDocInSession = useCallback(
    (docId: string) => {
      if (!session || !sessionId) return;
      const currentIds = session.doc_ids || [];
      const nextIds = currentIds.includes(docId)
        ? currentIds.filter(id => id !== docId)
        : [...currentIds, docId];
      
      patchSession.mutate({
        session_id: sessionId,
        body: { doc_ids: nextIds }
      });
    },
    [session, sessionId, patchSession]
  );

  return (
    <div className="h-full overflow-hidden grid grid-cols-[minmax(250px,20%)_minmax(400px,40%)_minmax(400px,40%)]">
      <DataPanel
        session={session}
        documents={globalDocuments}
        docsLoading={docsLoading}
        selectedDocId={selectedDoc?.id ?? null}
        onSelectDoc={handleSelectDoc}
        onToggleInSession={handleToggleDocInSession}
        onUpload={(file) => uploadDoc.mutate(file)}
        isUploading={uploadDoc.isPending}
        onDelete={(id) => deleteDoc.mutate(id)}
      />

      <ChatPanel
        sessionId={sessionId || ""}
        session={session ?? null}
      />

      <VisualPanel
        sessionId={sessionId || ""}
      />
    </div>
  );
}
