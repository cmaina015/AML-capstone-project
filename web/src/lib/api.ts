import { SERVER_MODE } from "./mode";
import type { Info, Prediction } from "./types";

/*
 * Model information is a JSON file written by scripts/export_static.py. Prediction runs in the browser
 * (lib/inference.ts), or on the FastAPI server when the site is built with NEXT_PUBLIC_INFERENCE=server.
 */

export async function fetchInfo(signal?: AbortSignal): Promise<Info> {
  const res = await fetch("/info.json", { signal });
  if (!res.ok) throw new Error(`Model information unavailable (HTTP ${res.status}).`);
  return res.json();
}

async function predictOnServer(file: File, explain: boolean): Promise<Prediction> {
  const body = new FormData();
  body.append("file", file);
  let res: Response;
  try {
    res = await fetch(`/api/predict?explain=${explain}`, { method: "POST", body });
  } catch {
    throw new Error("Could not reach the server. Check your connection and try again.");
  }
  if (!res.ok) {
    let detail: unknown;
    try {
      detail = (await res.json()).detail;
    } catch {
      /* not JSON */
    }
    throw new Error(typeof detail === "string" ? detail : `The server could not analyze the image (HTTP ${res.status}).`);
  }
  return res.json();
}

export async function predict(info: Info, file: File, explain: boolean): Promise<Prediction> {
  if (SERVER_MODE) return predictOnServer(file, explain);
  const { predictInBrowser } = await import("./inference");
  return predictInBrowser(info, file, explain);
}
