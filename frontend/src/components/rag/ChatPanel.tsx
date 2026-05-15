import { useState, useRef, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Send,
  Square,
  Bot,
  User,
  Trash2,
  Brain,
  RotateCcw,
  Sparkles,
  Loader2,
} from "lucide-react";
import { useQueryClient } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { MarkdownWithCitations } from "./MarkdownWithCitations";
import { ThinkingTimeline } from "./ThinkingTimeline";
import { useRAGChatStream } from "@/hooks/useRAGChatStream";
import { useChatHistory, useClearChatHistory } from "@/hooks/useChatHistory";
import { cn } from "@/lib/utils";
import type { Session, ChatMessage } from "@/types";

interface ChatPanelProps {
  sessionId: string;
  session: Session | null;
}

export function ChatPanel({ sessionId, session }: ChatPanelProps) {
  const [input, setInput] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);
  const queryClient = useQueryClient();

  const { data: history = [], isLoading: historyLoading } = useChatHistory(sessionId);
  const clearHistory = useClearChatHistory();

  const {
    stream,
    isStreaming,
    status,
    sendMessage,
    stopStream,
  } = useRAGChatStream(sessionId);

  // Refetch chat history when streaming ends so the persisted messages
  // appear immediately without requiring a page reload.
  const prevIsStreaming = useRef(false);
  useEffect(() => {
    if (prevIsStreaming.current && !isStreaming && stream.content) {
      queryClient.invalidateQueries({ queryKey: ["chat-history", sessionId] });
    }
    prevIsStreaming.current = isStreaming;
  }, [isStreaming, stream.content, queryClient, sessionId]);

  const scrollToBottom = useCallback(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [history, stream.content, stream.thinking, isStreaming, scrollToBottom]);

  const handleSend = (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!input.trim() || isStreaming) return;
    sendMessage(input);
    setInput("");
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="h-full flex flex-col bg-background relative overflow-hidden">
      {/* Messages List */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto px-4 py-6 space-y-8 scroll-smooth"
      >
        {historyLoading && history.length === 0 ? (
          <div className="h-full flex items-center justify-center">
            <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
          </div>
        ) : history.length === 0 && !isStreaming && !stream.content ? (
          <div className="h-full flex flex-col items-center justify-center text-center px-4">
            <div className="w-16 h-16 rounded-3xl bg-primary/10 flex items-center justify-center mb-6">
              <Sparkles className="w-8 h-8 text-primary" />
            </div>
            <h3 className="text-lg font-bold mb-2">Welcome to your Session</h3>
            <p className="text-sm text-muted-foreground max-w-sm">
              Upload documents to the left, then ask questions about them here.
              I'll provide citations for everything I say.
            </p>
          </div>
        ) : (
          <>
            {history.map((msg, idx) => (
              <div
                key={msg.id || idx}
                className={cn(
                  "flex gap-4 max-w-3xl mx-auto w-full",
                  msg.role === "user" ? "flex-row-reverse" : "flex-row"
                )}
              >
                <div className={cn(
                  "w-8 h-8 rounded-full flex-shrink-0 flex items-center justify-center shadow-sm",
                  msg.role === "user" ? "bg-primary text-primary-foreground" : "bg-card border"
                )}>
                  {msg.role === "user" ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
                </div>
                <div className={cn(
                  "flex-1 space-y-2",
                  msg.role === "user" ? "text-right" : "text-left"
                )}>
                  {msg.thinking && (
                    <ThinkingTimeline
                      steps={msg.agent_steps || []}
                      mode="embedded"
                      className="mb-4"
                    />
                  )}
                  <div className={cn(
                    "inline-block rounded-2xl px-4 py-3 text-sm shadow-sm",
                    msg.role === "user" 
                      ? "bg-primary text-primary-foreground" 
                      : "bg-card border text-foreground"
                  )}>
                    <MarkdownWithCitations 
                      content={msg.content} 
                      citations={msg.citations} 
                    />
                  </div>
                </div>
              </div>
            ))}

            {/* Current Stream */}
            {isStreaming && (
              <div className="flex gap-4 max-w-3xl mx-auto w-full">
                <div className="w-8 h-8 rounded-full bg-card border flex-shrink-0 flex items-center justify-center shadow-sm">
                  <Bot className="w-4 h-4" />
                </div>
                <div className="flex-1 space-y-4">
                  {stream.thinking && (
                    <ThinkingTimeline
                      steps={stream.steps}
                      mode="live"
                    />
                  )}
                  {stream.content && (
                    <div className="inline-block rounded-2xl px-4 py-3 text-sm bg-card border shadow-sm w-full">
                      <MarkdownWithCitations
                        content={stream.content}
                        citations={stream.citations}
                        isStreaming={true}
                      />
                    </div>
                  )}
                  {status === "streaming" && !stream.content && (
                    <div className="flex items-center gap-2 text-xs text-muted-foreground italic">
                      <Loader2 className="w-3 h-3 animate-spin" />
                      Thinking...
                    </div>
                  )}
                </div>
              </div>
            )}
          </>
        )}
      </div>

      {/* Input Area */}
      <div className="p-4 border-t bg-background/80 backdrop-blur-md">
        <div className="max-w-3xl mx-auto relative">
          <form
            onSubmit={handleSend}
            className="relative flex items-center"
          >
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a question..."
              className="w-full bg-muted/50 border-border/50 rounded-2xl pl-4 pr-12 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary/20 resize-none min-h-[50px] max-h-[200px]"
              rows={1}
            />
            <div className="absolute right-2 flex items-center gap-1">
              {isStreaming ? (
                <Button
                  type="button"
                  size="icon"
                  variant="ghost"
                  onClick={stopStream}
                  className="h-8 w-8 rounded-full text-destructive hover:bg-destructive/10"
                >
                  <Square className="w-4 h-4 fill-current" />
                </Button>
              ) : (
                <Button
                  type="submit"
                  size="icon"
                  disabled={!input.trim()}
                  className="h-8 w-8 rounded-full"
                >
                  <Send className="w-4 h-4" />
                </Button>
              )}
            </div>
          </form>
          <div className="mt-2 flex items-center justify-between px-2">
            <div className="flex items-center gap-4">
               <button
                onClick={() => sessionId && clearHistory.mutate(sessionId)}
                className="text-[10px] text-muted-foreground hover:text-destructive flex items-center gap-1 transition-colors"
                disabled={clearHistory.isPending}
              >
                <Trash2 className="w-3 h-3" />
                Clear Chat
              </button>
            </div>
            <p className="text-[10px] text-muted-foreground">
              Press Enter to send, Shift+Enter for new line
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
