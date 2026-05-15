import { useState, useCallback, useRef } from "react";
import { api, BASE_URL } from "@/lib/api";
import type { ChatMessage, Citation, AgentStep } from "@/types";

export type ChatStreamStatus = "idle" | "streaming" | "error";

export function useRAGChatStream(sessionId: string) {
  const [isStreaming, setIsStreaming] = useState(false);
  const [status, setStatus] = useState<ChatStreamStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  
  const [stream, setStream] = useState({
    content: "",
    thinking: "",
    citations: [] as Citation[],
    steps: [] as AgentStep[],
  });

  const abortRef = useRef<AbortController | null>(null);
  const readerRef = useRef<ReadableStreamDefaultReader | null>(null);
  const bufferRef = useRef("");
  const thinkingBufferRef = useRef("");
  const rafRef = useRef<number | undefined>(undefined);
  const thinkingRafRef = useRef<number | undefined>(undefined);

  const stopStream = useCallback(() => {
    abortRef.current?.abort();
    readerRef.current?.cancel().catch(() => {});
    setIsStreaming(false);
    setStatus("idle");
  }, []);

  const onToken = useCallback((text: string) => {
    bufferRef.current += text;
    if (!rafRef.current) {
      rafRef.current = requestAnimationFrame(() => {
        const chunk = bufferRef.current;
        bufferRef.current = "";
        rafRef.current = undefined;
        setStream((prev) => ({ ...prev, content: prev.content + chunk }));
      });
    }
  }, []);

  const onThinkingToken = useCallback((text: string) => {
    thinkingBufferRef.current += text;
    if (!thinkingRafRef.current) {
      thinkingRafRef.current = requestAnimationFrame(() => {
        const chunk = thinkingBufferRef.current;
        thinkingBufferRef.current = "";
        thinkingRafRef.current = undefined;
        setStream((prev) => {
          const idx = prev.steps.findIndex((s) => s.step === "analyzing");
          if (idx === -1) return prev;
          const updatedSteps = [...prev.steps];
          updatedSteps[idx] = {
            ...updatedSteps[idx],
            thinkingText: (updatedSteps[idx].thinkingText || "") + chunk,
          };
          return { ...prev, thinking: prev.thinking + chunk, steps: updatedSteps };
        });
      });
    }
  }, []);

  const sendMessage = useCallback(
    async (message: string) => {
      stopStream();

      // Cancel any pending RAF flushes and clear stale buffer state
      // from the previous response before starting a new one.
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = undefined;
      }
      if (thinkingRafRef.current) {
        cancelAnimationFrame(thinkingRafRef.current);
        thinkingRafRef.current = undefined;
      }
      bufferRef.current = "";
      thinkingBufferRef.current = "";

      setStream({
        content: "",
        thinking: "",
        citations: [],
        steps: [
          {
            id: "analyzing",
            step: "analyzing",
            detail: "Analyzing your question...",
            status: "active",
            timestamp: Date.now(),
          }
        ],
      });
      setError(null);
      setStatus("streaming");
      setIsStreaming(true);

      abortRef.current = new AbortController();
      try {
        const response = await fetch(
          `${BASE_URL}/sessions/${sessionId}/chat`,
          {
            method: "POST",
            headers: { 
              "Content-Type": "application/json",
              "Accept": "text/event-stream"
            },
            body: JSON.stringify({ query: message }),
            signal: abortRef.current.signal,
          },
        );

        if (!response.ok) {
          const err = await response.json().catch(() => ({ detail: "Stream request failed" }));
          throw new Error(err.detail || `Error: ${response.status}`);
        }

        const reader = response.body?.getReader();
        if (!reader) throw new Error("No response body");
        readerRef.current = reader;

        const decoder = new TextDecoder();
        let sseBuffer = "";
        let currentEventType = "unknown";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          sseBuffer += decoder.decode(value, { stream: true });
          const lines = sseBuffer.split("\n");
          sseBuffer = lines.pop() || "";

          for (const line of lines) {
            if (line.startsWith("event: ")) {
              currentEventType = line.slice(7).trim();
              continue;
            }
            if (line.startsWith("data: ")) {
              const jsonStr = line.slice(6).trim();
              if (!jsonStr) continue;

              let data: unknown;
              try {
                data = JSON.parse(jsonStr);
              } catch (e) {
                console.error("Error parsing SSE JSON:", e);
                currentEventType = "unknown";
                continue;
              }

              // Dispatch outside the JSON.parse try/catch so that
              // a thrown Error (e.g. from the "error" case) propagates
              // to the outer catch and sets React error state.
              const eventType = currentEventType;
              currentEventType = "unknown"; // reset before dispatch
              const d = data as Record<string, unknown>;
              switch (eventType) {
                case "citations": {
                  const citations = data as unknown as Citation[];
                  setStream(prev => ({
                    ...prev,
                    citations,
                    steps: [
                      ...prev.steps.map(s => s.step === "analyzing" ? { ...s, status: "completed" as const } : s),
                      {
                        id: "retrieving",
                        step: "sources_found",
                        detail: `Found ${citations.length} sources`,
                        status: "completed",
                        timestamp: Date.now(),
                        sourceCount: citations.length,
                      },
                      {
                        id: "generating",
                        step: "generating",
                        detail: "Generating answer...",
                        status: "active",
                        timestamp: Date.now(),
                      }
                    ]
                  }));
                  break;
                }
                case "thinking_delta":
                  onThinkingToken((d as { text?: string }).text || "");
                  break;
                case "content_delta":
                  onToken((d as { text?: string }).text || "");
                  break;
                case "done":
                  setIsStreaming(false);
                  setStatus("idle");
                  setStream(prev => ({
                    ...prev,
                    steps: prev.steps.map(s => s.status === "active" ? { ...s, status: "completed" as const } : s)
                  }));
                  break;
                case "error":
                  throw new Error((d as { detail?: string }).detail || "Stream error");
              }
            }
          }
        }
      } catch (err: any) {
        if (err.name === "AbortError") return;
        setError(err.message);
        setIsStreaming(false);
        setStatus("error");
      }
    },
    [sessionId, onToken, onThinkingToken, stopStream]
  );

  return {
    stream,
    isStreaming,
    status,
    error,
    sendMessage,
    stopStream,
  };
}
