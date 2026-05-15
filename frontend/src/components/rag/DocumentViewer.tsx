import { useState, useEffect, useRef, useMemo, useCallback, memo } from "react";
import { useQuery } from "@tanstack/react-query";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { FileText, List, ChevronRight, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { api } from "@/lib/api";
import type { Document, Citation } from "@/types";

interface Heading {
  id: string;
  text: string;
  level: number;
}

const TOCSidebar = memo(function TOCSidebar({
  headings,
  activeId,
  onSelect,
}: {
  headings: Heading[];
  activeId: string | null;
  onSelect: (id: string) => void;
}) {
  if (headings.length === 0) return null;
  return (
    <nav className="w-56 flex-shrink-0 border-r overflow-y-auto py-4 px-3 hidden lg:block bg-muted/10">
      <div className="flex items-center gap-2 px-2 mb-4">
        <List className="w-4 h-4 text-muted-foreground" />
        <span className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest">Contents</span>
      </div>
      <ul className="space-y-1">
        {headings.map((h) => (
          <li key={h.id}>
            <button
              onClick={() => onSelect(h.id)}
              className={cn(
                "w-full text-left text-xs py-1.5 px-2 rounded-lg transition-all truncate leading-tight",
                activeId === h.id
                  ? "text-primary font-semibold bg-primary/10 shadow-sm"
                  : "text-muted-foreground hover:bg-muted hover:text-foreground"
              )}
              style={{ paddingLeft: `${(h.level - 1) * 12 + 8}px` }}
              title={h.text}
            >
              {h.text}
            </button>
          </li>
        ))}
      </ul>
    </nav>
  );
});

function PageDivider({ pageNo }: { pageNo: number }) {
  return (
    <div className="flex items-center gap-3 py-8 select-none" data-page={pageNo}>
      <div className="flex-1 border-t border-dashed border-border/50" />
      <span className="text-[10px] font-bold text-muted-foreground/40 uppercase tracking-[0.2em]">
        Page {pageNo}
      </span>
      <div className="flex-1 border-t border-dashed border-border/50" />
    </div>
  );
}

function extractHeadings(markdown: string): Heading[] {
  const headings: Heading[] = [];
  const lines = markdown.split("\n");
  for (const line of lines) {
    const match = line.match(/^(#{1,4})\s+(.+)/);
    if (match) {
      const level = match[1].length;
      const text = match[2].replace(/[*_`#]/g, "").trim();
      const id = text.toLowerCase().replace(/[^a-z0-9\s-]/g, "").replace(/\s+/g, "-").slice(0, 80);
      headings.push({ id, text, level });
    }
  }
  return headings;
}

function getHeadingText(children: React.ReactNode): string {
  if (typeof children === "string") return children;
  if (Array.isArray(children)) return children.map(getHeadingText).join("");
  if (children && typeof children === "object" && "props" in children) {
    return getHeadingText((children as React.ReactElement<{ children?: React.ReactNode }>).props.children);
  }
  return String(children ?? "");
}

function generateHeadingId(text: string): string {
  return text.toLowerCase().replace(/[^a-z0-9\s-]/g, "").replace(/\s+/g, "-").slice(0, 80);
}

export const DocumentViewer = memo(function DocumentViewer({
  doc,
  highlightCitation,
  onScrolled,
}: {
  doc: Document;
  highlightCitation?: Citation | null;
  onScrolled?: () => void;
}) {
  const contentRef = useRef<HTMLDivElement>(null);
  const pageCounterRef = useRef(1);
  const [activeHeading, setActiveHeading] = useState<string | null>(null);
  const [showToc, setShowToc] = useState(true);

  const { data: markdown, isLoading, error } = useQuery({
    queryKey: ["document-markdown", doc.id],
    queryFn: () => api.getText(`/documents/${doc.id}/markdown`),
    enabled: doc.status === "indexed",
    staleTime: 5 * 60 * 1000,
  });

  const headings = useMemo(() => (markdown ? extractHeadings(markdown) : []), [markdown]);

  const mdComponents = useMemo<import("react-markdown").Components>(() => ({
    h1: ({ children, ...props }) => {
      const text = getHeadingText(children);
      const id = generateHeadingId(text);
      return <h1 id={id} {...props}>{children}</h1>;
    },
    h2: ({ children, ...props }) => {
      const text = getHeadingText(children);
      const id = generateHeadingId(text);
      return <h2 id={id} {...props}>{children}</h2>;
    },
    h3: ({ children, ...props }) => {
      const text = getHeadingText(children);
      const id = generateHeadingId(text);
      return <h3 id={id} {...props}>{children}</h3>;
    },
    h4: ({ children, ...props }) => {
      const text = getHeadingText(children);
      const id = generateHeadingId(text);
      return <h4 id={id} {...props}>{children}</h4>;
    },
    hr: () => {
      pageCounterRef.current += 1;
      return <PageDivider pageNo={pageCounterRef.current} />;
    },
    img: ({ src, alt, ...props }) => (
      <figure className="my-8 group">
        <img
          src={src}
          alt={alt || ""}
          loading="lazy"
          className="rounded-2xl max-w-full mx-auto border border-border/30 shadow-sm transition-transform group-hover:scale-[1.01]"
          style={{ minHeight: 120, objectFit: "contain", background: "rgba(0,0,0,0.02)" }}
          onLoad={(e) => { (e.target as HTMLImageElement).style.minHeight = "auto"; (e.target as HTMLImageElement).style.background = "none"; }}
          {...props}
        />
        {alt && <figcaption className="text-[11px] text-muted-foreground text-center mt-3 font-medium opacity-60">{alt}</figcaption>}
      </figure>
    ),
  }), []);

  useEffect(() => {
    if (!contentRef.current || headings.length === 0) return;
    const observer = new IntersectionObserver((entries) => {
      for (const entry of entries) if (entry.isIntersecting) setActiveHeading(entry.target.id);
    }, { root: contentRef.current, rootMargin: "-20% 0px -60% 0px", threshold: 0 });
    contentRef.current.querySelectorAll("h1, h2, h3, h4").forEach((el) => observer.observe(el));
    return () => observer.disconnect();
  }, [headings, markdown]);

  const scrollTo = useCallback((target: HTMLElement, block: "start" | "center" = "center") => {
    const container = contentRef.current;
    if (!container) return;
    const animate = (dest: number) => {
      const start = container.scrollTop;
      const dist = dest - start;
      const duration = Math.min(400, Math.abs(dist) * 0.5 + 150);
      const t0 = performance.now();
      const step = () => {
        const p = Math.min((performance.now() - t0) / duration, 1);
        const ease = 1 - Math.pow(1 - p, 3);
        container.scrollTop = start + dist * ease;
        if (p < 1) requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    };
    const getDest = () => {
      let offset = 0; let el: HTMLElement | null = target;
      while (el && el !== container) { offset += el.offsetTop; el = el.offsetParent as HTMLElement | null; }
      return block === "center" ? offset - container.clientHeight / 2 + target.offsetHeight / 2 : offset;
    };
    animate(getDest());
    setTimeout(() => animate(getDest()), 500);
  }, []);

  useEffect(() => {
    if (!contentRef.current || !markdown || !highlightCitation) return;
    const rafId = requestAnimationFrame(() => requestAnimationFrame(() => {
      const container = contentRef.current;
      if (!container) return;
      // Clear highlights
      container.querySelectorAll(".citation-hl").forEach(el => (el as HTMLElement).classList.remove("citation-hl", "bg-primary/10", "ring-2", "ring-primary/20", "rounded-xl"));

      // Find by page
      if (highlightCitation.page_num) {
        const el = container.querySelector(`[data-page="${highlightCitation.page_num}"]`) as HTMLElement;
        if (el) {
          scrollTo(el, "start");
          el.classList.add("citation-hl", "bg-primary/5", "rounded-xl", "transition-all", "duration-1000");
          setTimeout(() => el.classList.remove("citation-hl", "bg-primary/5"), 3000);
        }
      }
      onScrolled?.();
    }));
    return () => cancelAnimationFrame(rafId);
  }, [highlightCitation, markdown, scrollTo, onScrolled]);

  if (doc.status !== "indexed") return <div className="h-full flex items-center justify-center p-12 text-center text-muted-foreground"><div className="max-w-xs"><FileText className="w-12 h-12 mx-auto mb-4 opacity-20" /><p className="text-sm font-medium">Document is being processed</p><p className="text-xs mt-1 opacity-60">Please wait for indexing to complete to view the parsed content.</p></div></div>;
  if (isLoading) return <div className="h-full flex items-center justify-center"><Loader2 className="w-8 h-8 animate-spin text-muted-foreground/30" /></div>;
  if (error) return <div className="h-full flex items-center justify-center text-destructive p-8 text-center text-sm">Error loading document: {(error as Error).message}</div>;
  if (!markdown) return <div className="h-full flex items-center justify-center text-muted-foreground text-sm">No content available.</div>;

  return (
    <div className="flex h-full min-h-0 bg-background/50">
      <TOCSidebar headings={headings} activeId={activeHeading} onSelect={(id) => {
        const el = contentRef.current?.querySelector(`#${CSS.escape(id)}`) as HTMLElement;
        if (el) { el.scrollIntoView({ behavior: "smooth", block: "start" }); setActiveHeading(id); }
      }} />
      <div ref={contentRef} className="flex-1 min-h-0 overflow-y-auto scrollbar-thin scroll-smooth">
        <div className="max-w-[800px] mx-auto px-8 py-12">
          <header className="mb-12 pb-8 border-b">
            <h1 className="text-3xl font-bold tracking-tight mb-4">{doc.original_filename}</h1>
            <div className="flex flex-wrap gap-4 text-xs font-medium text-muted-foreground uppercase tracking-widest opacity-60">
              {doc.page_count && <span>{doc.page_count} Pages</span>}
              {doc.chunk_count > 0 && <span>{doc.chunk_count} Chunks</span>}
              <span>{doc.parser_version || "Smart Parser"}</span>
            </div>
          </header>
          <article className="prose prose-slate dark:prose-invert max-w-none prose-headings:scroll-mt-10">
            <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]} components={mdComponents}>
              {markdown}
            </ReactMarkdown>
          </article>
        </div>
      </div>
    </div>
  );
});
