import type { Metadata } from "next";
import { Nav } from "@/components/Nav";
import { SERVER_MODE } from "@/lib/mode";
import "./globals.css";

export const metadata: Metadata = {
  title: { default: "Breast Histopathology Classifier", template: "%s · Breast Histopathology Classifier" },
  description: "Swin Tiny model that classifies BreakHis 100X H&E images as benign or malignant, with Grad-CAM.",
  icons: { icon: "/icon.svg" },
};

// Runs before first paint so the page never flashes the wrong theme: saved choice, else OS setting.
const THEME_SCRIPT = `(function(){try{var t=localStorage.getItem("theme");var d=t?t==="dark":matchMedia("(prefers-color-scheme: dark)").matches;document.documentElement.classList.toggle("dark",d)}catch(e){}})()`;

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }} />
      </head>
      <body className="flex min-h-screen flex-col antialiased">
        <Nav />
        <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-8">{children}</main>
        <footer className="border-t border-line bg-surface">
          <div className="mx-auto max-w-6xl px-4 py-4 text-xs text-fg-muted">
            For research and education only. Not intended for clinical diagnosis. Data:{" "}
            <a
              className="underline hover:text-fg-2"
              href="https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/"
              target="_blank"
              rel="noopener"
            >
              BreakHis
            </a>
            .{" "}
            {SERVER_MODE ? (
              <>
                Served by FastAPI.{" "}
                <a className="underline hover:text-fg-2" href="/docs">
                  API docs
                </a>
                . Uploaded images are not stored.
              </>
            ) : (
              "Images are analyzed in your browser and are not uploaded."
            )}
          </div>
        </footer>
      </body>
    </html>
  );
}
