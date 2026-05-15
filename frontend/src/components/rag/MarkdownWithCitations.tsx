import { useState, useMemo, memo } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { PrismLight as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark, oneLight } from "react-syntax-highlighter/dist/esm/styles/prism";
import { Copy, ClipboardCheck, FileText, Brain } from "lucide-react";
import { useThemeStore } from "@/stores/useThemeStore";
import { useSessionStore } from "@/stores/useSessionStore";
import { StreamingMarkdown } from "./MemoizedMarkdown";
import { cn } from "@/lib/utils";
import type { Citation } from "@/types";

// Syntax highlighting registration (subset)
import python from "react-syntax-highlighter/dist/esm/languages/prism/python";
import javascript from "react-syntax-highlighter/dist/esm/languages/prism/javascript";
import typescript from "react-syntax-highlighter/dist/esm/languages/prism/typescript";
import bash from "react-syntax-highlighter/dist/esm/languages/prism/bash";
import json from "react-syntax-highlighter/dist/esm/languages/prism/json";
import sql from "react-syntax-highlighter/dist/esm/languages/prism/sql";
import markdown from "react-syntax-highlighter/dist/esm/languages/prism/markdown";

SyntaxHighlighter.registerLanguage("python", python);
SyntaxHighlighter.registerLanguage("javascript", javascript);
SyntaxHighlighter.registerLanguage("js", javascript);
SyntaxHighlighter.registerLanguage("typescript", typescript);
SyntaxHighlighter.registerLanguage("ts", typescript);
SyntaxHighlighter.registerLanguage("bash", bash);
SyntaxHighlighter.registerLanguage("sh", bash);
SyntaxHighlighter.registerLanguage("json", json);
SyntaxHighlighter.registerLanguage("sql", sql);
SyntaxHighlighter.registerLanguage("markdown", markdown);
SyntaxHighlighter.registerLanguage("md", markdown);

function CitationLink({ index, citation }: { index: string; citation: Citation }) {
  const { setHighlight } = useSessionStore();
  return (
    <button
      onClick={() => setHighlight(citation)}
      className="inline-flex items-center gap-0.5 h-[18px] px-1.5 mx-0.5 text-[10px] font-medium rounded-full bg-primary/10 text-primary hover:bg-primary/20 transition-colors align-middle"
    >
      <FileText className="w-2.5 h-2.5" />
      <span>{index}</span>
    </button>
  );
}

const CITATION_RE = /\[(\d+)\]/g;

function CodeBlock({ language, value }: { language: string; value: string }) {
  const [copied, setCopied] = useState(false);
  const theme = useThemeStore((s) => s.theme);
  const isDark = theme === "dark";

  const handleCopy = () => {
    navigator.clipboard.writeText(value).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  return (
    <div className="group relative my-4">
      <div className="absolute right-2 top-2 z-10 flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
        <span className="text-[10px] font-mono text-muted-foreground uppercase">{language}</span>
        <button
          onClick={handleCopy}
          className="p-1.5 rounded-md bg-background/50 hover:bg-background border border-border/50 transition-colors"
        >
          {copied ? <ClipboardCheck className="w-3.5 h-3.5 text-emerald-500" /> : <Copy className="w-3.5 h-3.5" />}
        </button>
      </div>
      <SyntaxHighlighter
        language={language}
        style={isDark ? oneDark : oneLight}
        customStyle={{
          margin: 0,
          borderRadius: "12px",
          fontSize: "13px",
          padding: "16px",
          background: isDark ? "rgba(0,0,0,0.2)" : "rgba(0,0,0,0.02)",
          border: "1px solid var(--border)",
        }}
      >
        {value}
      </SyntaxHighlighter>
    </div>
  );
}

export const MarkdownWithCitations = memo(function MarkdownWithCitations({
  content,
  citations = [],
  isStreaming = false,
}: {
  content: string;
  citations?: Citation[];
  isStreaming?: boolean;
}) {
  const renderBlock = (blockContent: string) => (
    <ReactMarkdown
      remarkPlugins={[remarkGfm, remarkMath]}
      rehypePlugins={[rehypeKatex]}
      components={{
        p: ({ children }) => {
          if (typeof children === "string") {
            const parts = children.split(CITATION_RE);
            return (
              <p className="mb-4 last:mb-0 leading-relaxed">
                {parts.map((part, i) => {
                  if (i % 2 === 1) {
                    const citation = citations.find((c) => String(c.index) === part);
                    return citation ? (
                      <CitationLink key={i} index={part} citation={citation} />
                    ) : (
                      `[${part}]`
                    );
                  }
                  return part;
                })}
              </p>
            );
          }
          return <p className="mb-4 last:mb-0 leading-relaxed">{children}</p>;
        },
        code: ({ className, children, ...props }: any) => {
          const match = /language-(\w+)/.exec(className || "");
          const isInline = !match;
          if (isInline) {
            return (
              <code className="px-1.5 py-0.5 rounded-md bg-muted font-mono text-[0.9em]" {...props}>
                {children}
              </code>
            );
          }
          return <CodeBlock language={match[1]} value={String(children).replace(/\n$/, "")} />;
        },
        table: ({ children }) => (
          <div className="my-6 overflow-x-auto rounded-xl border border-border/50 bg-background/50">
            <table className="w-full text-sm border-collapse">{children}</table>
          </div>
        ),
        th: ({ children }) => (
          <th className="px-4 py-3 text-left font-semibold border-b border-border/50 bg-muted/30">{children}</th>
        ),
        td: ({ children }) => <td className="px-4 py-2 border-b border-border/50">{children}</td>,
      }}
    >
      {blockContent}
    </ReactMarkdown>
  );

  return (
    <div className="prose prose-sm dark:prose-invert max-w-none">
      <StreamingMarkdown
        content={content}
        isStreaming={isStreaming}
        renderBlock={renderBlock}
      />
    </div>
  );
});
