"use client";
import { usePathname } from "next/navigation";
import { ThemeToggle } from "./ThemeToggle";

const LINKS = [
  { href: "/", label: "Analyze" },
  { href: "/pipeline/index.html", label: "Pipeline" },
  { href: "/data/index.html", label: "Data" },
  { href: "/slides/index.html", label: "Slides" },
];

export function Nav() {
  const path = usePathname();
  // Links name the file explicitly (/pipeline/index.html): Hugging Face static Spaces do not serve a
  // folder's index.html for /pipeline/. Compare the first path segment so either URL form matches.
  const section = (p: string) => (p.split("/")[1] ?? "").replace("index.html", "");
  const active = (href: string) => section(path ?? "/") === section(href);
  return (
    <header className="border-t-[3px] border-b border-t-brand border-b-line bg-surface">
      <div className="mx-auto flex h-14 max-w-6xl items-center justify-between gap-2 px-4 sm:gap-4">
        <a href="/" className="whitespace-nowrap font-semibold text-fg">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src="/icon.svg" alt="Breast Histopathology Classifier" width={28} height={28} className="sm:hidden" />
          <span className="hidden sm:inline">Breast Histopathology Classifier</span>
        </a>
        <div className="flex items-center gap-2">
          <nav className="flex gap-1 text-sm">
            {LINKS.map((l) => (
              <a
                key={l.href}
                href={l.href}
                className={`rounded-md px-2 py-1.5 sm:px-3 ${
                  active(l.href) ? "bg-brand-light font-medium text-brand-fg" : "text-fg-muted hover:text-brand-fg"
                }`}
              >
                {l.label}
              </a>
            ))}
          </nav>
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
}
