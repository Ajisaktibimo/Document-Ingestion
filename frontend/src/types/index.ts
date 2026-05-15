export interface Session {
  session_id: string;
  title: string;
  doc_ids: string[];
  created_at: string;
  updated_at: string;
}

export interface Document {
  id: string;
  original_filename: string;
  status: DocumentStatus;
  created_at: string;
  updated_at: string;
  page_count?: number;
  chunk_count: number;
  parser_version?: string;
}

export type DocumentStatus = "pending" | "parsing" | "indexing" | "processing" | "indexed" | "failed";

export interface Citation {
  index: number | string;
  id: string;
  doc_id: string;
  heading?: string;
  heading_path?: string[];
  page_num?: number;
  score: number;
}

export interface ChatImageRef {
  image_id: string;
  ref_id: string;
  document_id: string;
  page_no: number;
  url: string;
  caption?: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  thinking?: string | null;
  created_at: string;
  agent_steps?: AgentStep[];
}

export type AgentStepType = "analyzing" | "understood" | "retrieving" | "sources_found" | "generating" | "done" | "error";

export interface AgentStep {
  id: string;
  step: AgentStepType;
  detail: string;
  status: "active" | "completed" | "error";
  timestamp: number;
  durationMs?: number;
  thinkingText?: string;
  sourceCount?: number;
}
