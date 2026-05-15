import { create } from "zustand";
import type { Document, Citation } from "@/types";

interface SessionState {
  selectedDoc: Document | null;
  highlight: Citation | null;
  
  selectDoc: (doc: Document | null) => void;
  setHighlight: (citation: Citation | null, doc?: Document) => void;
  reset: () => void;
}

export const useSessionStore = create<SessionState>((set) => ({
  selectedDoc: null,
  highlight: null,

  selectDoc: (doc) => {
    set({
      selectedDoc: doc,
      highlight: null,
    });
  },

  setHighlight: (citation, doc) => {
    set((state) => ({
      selectedDoc: doc || state.selectedDoc,
      highlight: citation,
    }));
  },

  reset: () => {
    set({
      selectedDoc: null,
      highlight: null,
    });
  },
}));
