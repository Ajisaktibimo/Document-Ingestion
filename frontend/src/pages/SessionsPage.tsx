import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";
import { useSessions, useCreateSession, useDeleteSession } from "@/hooks/useSessions";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  Plus,
  MessageSquare,
  Trash2,
  MoreHorizontal,
  X,
  History,
} from "lucide-react";
import { ConfirmDialog } from "@/components/ui/confirm-dialog";
import type { Session } from "@/types";

export function SessionsPage() {
  const navigate = useNavigate();
  const { data: sessions, isLoading } = useSessions();
  const createSession = useCreateSession();
  const deleteSession = useDeleteSession();
  const [showNewSession, setShowNewSession] = useState(false);
  const [newSessionTitle, setNewSessionTitle] = useState("");
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);
  const [openMenu, setOpenMenu] = useState<string | null>(null);

  const handleCreateSession = async () => {
    if (!newSessionTitle.trim()) return;
    try {
      const s = await createSession.mutateAsync({ title: newSessionTitle });
      toast.success("Chat session created");
      setNewSessionTitle("");
      setShowNewSession(false);
      navigate(`/sessions/${s.session_id}`);
    } catch {
      toast.error("Failed to create session");
    }
  };

  const handleDeleteSession = async (id: string) => {
    try {
      await deleteSession.mutateAsync(id);
      toast.success("Session deleted");
    } catch {
      toast.error("Failed to delete session");
    }
    setDeleteConfirm(null);
  };

  const formatDate = (dateStr: string) => {
    try {
      const date = new Date(dateStr);
      const now = new Date();
      const diff = now.getTime() - date.getTime();
      const days = Math.floor(diff / (1000 * 60 * 60 * 24));
      if (days === 0) return "Today";
      if (days === 1) return "Yesterday";
      if (days < 7) return `${days} days ago`;
      return date.toLocaleDateString("en-US", { month: "short", day: "numeric" });
    } catch {
      return dateStr;
    }
  };

  return (
    <div className="h-full overflow-y-auto bg-background/50">
      <div className="max-w-5xl mx-auto px-6 py-8">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h2 className="text-2xl font-bold tracking-tight">Your Chats</h2>
            <p className="text-sm text-muted-foreground mt-1">
              {sessions?.length || 0} active session{sessions?.length !== 1 ? "s" : ""}
            </p>
          </div>
          <Button onClick={() => setShowNewSession(true)} className="rounded-full shadow-lg hover:shadow-primary/20 transition-all">
            <Plus className="w-4 h-4 mr-1.5" />
            New Chat
          </Button>
        </div>

        {showNewSession && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-md animate-in fade-in duration-300">
            <Card className="w-full max-w-md mx-4 shadow-2xl border-primary/20">
              <CardContent className="pt-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-xl font-bold">Start New Chat</h3>
                  <button
                    onClick={() => setShowNewSession(false)}
                    className="w-8 h-8 flex items-center justify-center rounded-full text-muted-foreground hover:bg-muted transition-colors"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
                <Input
                  placeholder="Session title (e.g., Financial Report Analysis)"
                  value={newSessionTitle}
                  onChange={(e) => setNewSessionTitle(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleCreateSession()}
                  autoFocus
                  className="text-lg py-6"
                />
                <div className="flex justify-end gap-3 mt-6">
                  <Button variant="ghost" onClick={() => setShowNewSession(false)}>
                    Cancel
                  </Button>
                  <Button onClick={handleCreateSession} disabled={createSession.isPending || !newSessionTitle.trim()}>
                    {createSession.isPending ? "Starting..." : "Start Chat"}
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {isLoading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {[1, 2, 3].map((i) => (
              <Card key={i} className="animate-pulse h-32 bg-muted/30" />
            ))}
          </div>
        ) : !sessions || sessions.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-20 text-center">
            <div className="w-24 h-24 rounded-3xl bg-primary/10 flex items-center justify-center mb-6 animate-bounce-slow">
              <MessageSquare className="w-12 h-12 text-primary" />
            </div>
            <h3 className="text-2xl font-bold mb-3">No chat history yet</h3>
            <p className="text-muted-foreground max-w-sm mb-8 leading-relaxed">
              Start a new chat to analyze documents, ask questions, and get insights with AI.
            </p>
            <Button onClick={() => setShowNewSession(true)} size="lg" className="rounded-full px-8">
              <Plus className="w-4 h-4 mr-2" />
              Create First Session
            </Button>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {sessions.map((s: Session) => (
              <Card
                key={s.session_id}
                className="group relative cursor-pointer overflow-hidden transition-all hover:ring-2 hover:ring-primary/40 hover:-translate-y-1 hover:shadow-xl bg-card/80 backdrop-blur-sm border-border/50"
                onClick={() => navigate(`/sessions/${s.session_id}`)}
              >
                <CardContent className="p-5">
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center gap-3 min-w-0">
                      <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center flex-shrink-0 group-hover:bg-primary group-hover:text-primary-foreground transition-all duration-300">
                        <MessageSquare className="w-5 h-5" />
                      </div>
                      <div className="min-w-0">
                        <h3 className="font-bold text-base truncate pr-6">{s.title}</h3>
                        <p className="text-xs text-muted-foreground flex items-center gap-1.5 mt-0.5">
                          <History className="w-3 h-3" />
                          {formatDate(s.updated_at)}
                        </p>
                      </div>
                    </div>
                    <div className="absolute top-4 right-2">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setOpenMenu(openMenu === s.session_id ? null : s.session_id);
                        }}
                        className="w-8 h-8 flex items-center justify-center rounded-full text-muted-foreground opacity-0 group-hover:opacity-100 hover:bg-muted transition-all"
                      >
                        <MoreHorizontal className="w-4 h-4" />
                      </button>
                      {openMenu === s.session_id && (
                        <div className="absolute right-0 top-10 z-20 bg-card border border-border/50 rounded-xl shadow-2xl py-1.5 w-36 animate-in fade-in zoom-in-95">
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              setDeleteConfirm(s.session_id);
                              setOpenMenu(null);
                            }}
                            className="w-full flex items-center gap-2.5 px-3 py-2 text-sm text-destructive hover:bg-destructive/10 transition-colors"
                          >
                            <Trash2 className="w-4 h-4" />
                            Delete Session
                          </button>
                        </div>
                      )}
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="text-[10px] px-2 py-0.5 rounded-full bg-secondary text-secondary-foreground font-medium uppercase tracking-wider">
                      {s.message_count} messages
                    </div>
                    {s.doc_ids && s.doc_ids.length > 0 && (
                      <div className="text-[10px] px-2 py-0.5 rounded-full bg-primary/5 text-primary font-medium uppercase tracking-wider">
                        {s.doc_ids.length} docs
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </div>

      <ConfirmDialog
        open={deleteConfirm !== null}
        onConfirm={() => deleteConfirm !== null && handleDeleteSession(deleteConfirm)}
        onCancel={() => setDeleteConfirm(null)}
        title="Delete Session"
        message="This will permanently delete this chat session and all its messages. This action cannot be undone."
        confirmLabel="Delete Permanently"
        variant="danger"
      />
    </div>
  );
}
