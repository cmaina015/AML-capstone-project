import type { Metadata } from "next";
import { PipelinePage } from "@/components/PipelinePage";

export const metadata: Metadata = { title: "Pipeline" };

export default function Page() {
  return <PipelinePage />;
}
