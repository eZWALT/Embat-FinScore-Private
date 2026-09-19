import { type ReactNode } from "react";

/**
 * Safe chat markdown: paragraphs, bold, italic, inline code, fenced blocks,
 * line breaks, hyphen lists. No raw HTML, no images, https links only.
 */
export function AgentMarkdown({
  text,
  streaming = false,
}: {
  text: string;
  streaming?: boolean;
}) {
  if (!text) return null;
  const blocks = splitBlocks(text.replace(/\r\n/g, "\n"));

  return (
    <div
      className={
        streaming
          ? "agent-md agent-md-streaming min-w-0 max-w-full space-y-2 overflow-x-hidden text-sm leading-snug"
          : "agent-md min-w-0 max-w-full space-y-2 overflow-x-hidden text-sm leading-snug"
      }
    >
      {blocks.map((block, index) => {
        if (block.type === "code") {
          return (
            <pre
              key={`c-${index}`}
              className="max-w-full overflow-x-auto whitespace-pre-wrap rounded-md border bg-muted/50 p-2.5 font-mono text-[12px] leading-snug wrap-break-word"
            >
              <code className="block max-w-full font-mono wrap-break-word">{block.text}</code>
            </pre>
          );
        }
        if (block.type === "ul") {
          return (
            <ul key={`u-${index}`} className="min-w-0 max-w-full list-disc space-y-1 pl-4">
              {block.items.map((item, itemIndex) => (
                <li key={`i-${itemIndex}`} className="min-w-0 max-w-full wrap-break-word">
                  {renderInline(item, `u${index}-${itemIndex}`)}
                </li>
              ))}
            </ul>
          );
        }
        return (
          <p key={`p-${index}`} className="min-w-0 max-w-full wrap-break-word">
            {renderParagraph(block.text, `p${index}`)}
          </p>
        );
      })}
    </div>
  );
}

type Block = { type: "p"; text: string } | { type: "ul"; items: string[] } | { type: "code"; text: string };

function splitBlocks(src: string): Block[] {
  const blocks: Block[] = [];
  const lines = src.split("\n");
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    if (line.startsWith("```")) {
      const body: string[] = [];
      i += 1;
      while (i < lines.length && !lines[i].startsWith("```")) {
        body.push(lines[i]);
        i += 1;
      }
      if (i < lines.length) i += 1;
      blocks.push({ type: "code", text: body.join("\n") });
      continue;
    }

    if (/^-\s+/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^-\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^-\s+/, ""));
        i += 1;
      }
      blocks.push({ type: "ul", items });
      continue;
    }

    if (line.trim() === "") {
      i += 1;
      continue;
    }

    const para: string[] = [];
    while (
      i < lines.length &&
      lines[i].trim() !== "" &&
      !lines[i].startsWith("```") &&
      !/^-\s+/.test(lines[i])
    ) {
      para.push(lines[i]);
      i += 1;
    }
    blocks.push({ type: "p", text: para.join("\n") });
  }

  return blocks;
}

function renderParagraph(text: string, keyPrefix: string): ReactNode[] {
  const lines = text.split("\n");
  const nodes: ReactNode[] = [];
  lines.forEach((line, lineIndex) => {
    if (lineIndex > 0) nodes.push(<br key={`${keyPrefix}-br-${lineIndex}`} />);
    nodes.push(...renderInline(line, `${keyPrefix}-${lineIndex}`));
  });
  return nodes;
}

function httpsHref(raw: string): string | null {
  const trimmed = raw.trim();
  if (!/^https:\/\//i.test(trimmed)) return null;
  try {
    const url = new URL(trimmed);
    return url.protocol === "https:" ? url.href : null;
  } catch {
    return null;
  }
}

function renderInline(input: string, keyPrefix: string): ReactNode[] {
  const out: ReactNode[] = [];
  let i = 0;
  let buf = "";
  let n = 0;

  const flush = () => {
    if (!buf) return;
    out.push(buf);
    buf = "";
  };

  const push = (node: ReactNode) => {
    flush();
    out.push(node);
  };

  while (i < input.length) {
    if (input[i] === "`") {
      const end = input.indexOf("`", i + 1);
      if (end !== -1) {
        push(
          <code
            key={`${keyPrefix}-code-${n++}`}
            className="wrap-break-word rounded bg-muted px-1 py-0.5 font-mono text-[0.85em]"
          >
            {input.slice(i + 1, end)}
          </code>,
        );
        i = end + 1;
        continue;
      }
    }

    if (input.startsWith("![", i)) {
      const img = input.slice(i).match(/^!\[([^\]]*)\]\(([^)]*)\)/);
      if (img) {
        buf += img[1];
        i += img[0].length;
        continue;
      }
    }

    if (input[i] === "[") {
      const link = input.slice(i).match(/^\[([^\]]+)\]\(([^)]+)\)/);
      if (link) {
        const href = httpsHref(link[2]);
        if (href) {
          push(
            <a
              key={`${keyPrefix}-a-${n++}`}
              href={href}
              rel="noopener noreferrer"
              target="_blank"
              className="underline underline-offset-2"
            >
              {renderInline(link[1], `${keyPrefix}-a${n}`)}
            </a>,
          );
        } else {
          buf += link[1];
        }
        i += link[0].length;
        continue;
      }
    }

    if (input.startsWith("**", i)) {
      const end = input.indexOf("**", i + 2);
      if (end !== -1) {
        push(
          <strong key={`${keyPrefix}-b-${n++}`}>
            {renderInline(input.slice(i + 2, end), `${keyPrefix}-b${n}`)}
          </strong>,
        );
        i = end + 2;
        continue;
      }
    }

    if (input[i] === "*") {
      const end = input.indexOf("*", i + 1);
      if (end !== -1 && input[end + 1] !== "*") {
        push(
          <em key={`${keyPrefix}-em-${n++}`}>
            {renderInline(input.slice(i + 1, end), `${keyPrefix}-em${n}`)}
          </em>,
        );
        i = end + 1;
        continue;
      }
    }

    buf += input[i];
    i += 1;
  }

  flush();
  return out;
}
