"use client";
import { useEffect, useState } from "react";
import { fetchInfo } from "./api";
import type { Info } from "./types";

let cache: Info | null = null;
let pending: Promise<Info> | null = null;

/** /info.json (model card, metrics, dataset stats), fetched once and shared across pages. */
export function useInfo() {
  const [info, setInfo] = useState<Info | null>(cache);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    if (cache) return;
    pending ??= fetchInfo();
    pending
      .then((i) => {
        cache = i;
        setInfo(i);
      })
      .catch((e: Error) => {
        pending = null;
        setError(e.message);
      });
  }, []);
  return { info, error };
}
