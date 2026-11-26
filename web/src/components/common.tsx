import type { ReactNode } from "react";

export function PageHeader({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="mb-6">
      <h1 className="text-2xl font-semibold text-fg">{title}</h1>
      {children && <p className="mt-1 max-w-3xl text-sm text-fg-muted">{children}</p>}
    </div>
  );
}

export function Card({ title, children, className = "" }: { title?: string; children: ReactNode; className?: string }) {
  return (
    <section className={`rounded-lg border border-line bg-surface p-5 ${className}`}>
      {title && <h2 className="mb-3 text-base font-semibold text-fg">{title}</h2>}
      {children}
    </section>
  );
}

export function Section({ id, title, children }: { id?: string; title: string; children: ReactNode }) {
  return (
    <section id={id} className="border-t border-line py-8 first:border-t-0 first:pt-0">
      <h2 className="mb-3 text-lg font-semibold text-fg">{title}</h2>
      <div className="space-y-4 text-sm leading-relaxed text-fg-2">{children}</div>
    </section>
  );
}

export function Table({
  head,
  rows,
  align,
  className = "",
}: {
  head: ReactNode[];
  rows: ReactNode[][];
  align?: ("left" | "right")[];
  className?: string;
}) {
  const cls = (i: number) => (align?.[i] === "right" ? "text-right tabular" : "text-left");
  return (
    <div className={`overflow-x-auto rounded-md border border-line bg-surface ${className}`}>
      <table className="w-full text-sm">
        <thead className="bg-subtle text-xs text-fg-muted">
          <tr>
            {head.map((h, i) => (
              <th key={i} className={`px-3 py-2 font-medium ${cls(i)}`}>
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-line-soft">
          {rows.map((r, i) => (
            <tr key={i}>
              {r.map((c, j) => (
                <td key={j} className={`px-3 py-2 text-fg-2 ${cls(j)}`}>
                  {c}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function Figure({ src, alt, caption, className = "" }: { src: string; alt: string; caption?: ReactNode; className?: string }) {
  return (
    <figure className={className}>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={src} alt={alt} className="w-full rounded-md border border-line bg-white dark:brightness-[.88]" />
      {caption && <figcaption className="mt-1.5 text-xs text-fg-muted">{caption}</figcaption>}
    </figure>
  );
}

export function Note({ children, tone = "gray" }: { children: ReactNode; tone?: "gray" | "yellow" }) {
  const c = tone === "yellow" ? "border-warn-line bg-warn text-warn-fg" : "border-line bg-subtle text-fg-2";
  return <div className={`rounded-md border px-3 py-2 text-sm ${c}`}>{children}</div>;
}

export function Loading() {
  return <p className="text-sm text-fg-muted">Loading…</p>;
}
