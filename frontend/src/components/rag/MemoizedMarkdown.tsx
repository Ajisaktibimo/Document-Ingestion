import { memo, useMemo } from "react";
import type { Citation, ChatImageRef } from "@/types";

interface SplitResult {
  completed: string[];
  inProgress: string;
}

export function splitIntoBlocks(content: string): SplitResult {
  const lines = content.split("\n");
  const blocks: string[] = [];
  let currentBlock: string[] = [];
  let inCodeFence = false;
  let inLatexBlock = false;

  for (const line of lines) {
    if (line.trimStart().startsWith("```")) {
      inCodeFence = !inCodeFence;
    }
    if (!inCodeFence) {
      const trimmed = line.trim();
      if (trimmed === "$$" || (trimmed.startsWith("$$") && !trimmed.endsWith("$$"))) {
        inLatexBlock = !inLatexBlock;
      } else if (trimmed.endsWith("$$") && inLatexBlock) {
        inLatexBlock = false;
      }
    }
    if (line.trim() === "" && !inCodeFence && !inLatexBlock) {
      if (currentBlock.length > 0) {
        blocks.push(currentBlock.join("\n"));
        currentBlock = [];
      }
      continue;
    }
    currentBlock.push(line);
  }
  if (inCodeFence || inLatexBlock || currentBlock.length > 0) {
    const inProgress = currentBlock.join("\n");
    return { completed: blocks, inProgress };
  }
  return { completed: blocks, inProgress: "" };
}

export function sanitizeInProgress(text: string): string {
  if (!text) return "";
  let result = text;
  const latexCount = (result.match(/\$\$/g) || []).length;
  if (latexCount % 2 !== 0) {
    const lastIdx = result.lastIndexOf("$$");
    const afterDollars = result.slice(lastIdx + 2);
    if (afterDollars.trim()) {
      result = result.slice(0, lastIdx) + "$$\n" + afterDollars.trimStart() + "\n$$";
    } else {
      result = result + "\n$$";
    }
  }
  const fenceCount = (result.match(/```/g) || []).length;
  if (fenceCount % 2 !== 0) {
    result = result + "\n```";
  }
  const lines = result.split("\n");
  while (lines.length > 0) {
    const last = lines[lines.length - 1];
    if (last.startsWith("|") && !last.trimEnd().endsWith("|")) {
      lines.pop();
    } else {
      break;
    }
  }
  result = lines.join("\n");
  return result.trimEnd();
}

function stableHash(str: string): string {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const ch = str.charCodeAt(i);
    hash = ((hash << 5) - hash + ch) | 0;
  }
  return hash.toString(36);
}

interface MemoizedBlockProps {
  content: string;
  renderFn: (content: string) => React.ReactNode;
}

const MemoizedMarkdownBlock = memo(
  function MemoizedMarkdownBlock({ content, renderFn }: MemoizedBlockProps) {
    return <>{renderFn(content)}</>;
  },
  (prev, next) => prev.content === next.content && prev.renderFn === next.renderFn
);

export interface StreamingMarkdownProps {
  content: string;
  sources?: Citation[];
  imageRefs?: ChatImageRef[];
  isStreaming?: boolean;
  renderBlock: (content: string) => React.ReactNode;
}

export function StreamingMarkdown({
  content,
  isStreaming = false,
  renderBlock,
}: StreamingMarkdownProps) {
  const { completed, inProgress } = useMemo(
    () => splitIntoBlocks(content),
    [content]
  );

  const sanitized = useMemo(
    () => (isStreaming ? sanitizeInProgress(inProgress) : inProgress),
    [inProgress, isStreaming]
  );

  return (
    <>
      {completed.map((block, i) => (
        <MemoizedMarkdownBlock
          key={`b-${stableHash(block)}-${i}`}
          content={block}
          renderFn={renderBlock}
        />
      ))}
      {sanitized ? renderBlock(sanitized) : null}
    </>
  );
}
