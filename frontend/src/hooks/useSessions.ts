import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { Session } from "@/types";

export function useSessions() {
  return useQuery({
    queryKey: ["sessions"],
    queryFn: () => api.get<Session[]>("/sessions"),
  });
}

export function useSession(sessionId: string | null) {
  return useQuery({
    queryKey: ["sessions", sessionId],
    queryFn: () => api.get<Session>(`/sessions/${sessionId}`),
    enabled: !!sessionId,
  });
}

export function useCreateSession() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (data: { title?: string; docIds?: string[] } = {}) =>
      api.post<Session>("/sessions", { title: data.title, doc_ids: data.docIds || [] }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["sessions"] });
    },
  });
}

export function usePatchSession() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ session_id, body }: { session_id: string; body: Partial<Session> }) =>
      api.patch<Session>(`/sessions/${session_id}`, body),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["sessions"] });
    },
  });
}

export function useDeleteSession() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api.delete(`/sessions/${id}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["sessions"] });
    },
  });
}
