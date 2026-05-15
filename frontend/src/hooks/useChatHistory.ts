import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { ChatMessage } from "@/types";

export function useChatHistory(sessionId: string | undefined) {
  return useQuery({
    queryKey: ["chat-history", sessionId],
    queryFn: () => {
      if (!sessionId) return [];
      return api.get<ChatMessage[]>(`/sessions/${sessionId}/messages`);
    },
    enabled: !!sessionId,
    staleTime: 1000 * 60 * 5, // 5 minutes
  });
}

export function useClearChatHistory() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (sessionId: string) => api.delete(`/sessions/${sessionId}/messages`),
    onSuccess: (_, sessionId) => {
      queryClient.setQueryData(["chat-history", sessionId], []);
    },
  });
}
