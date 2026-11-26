import type { Metadata } from "next";
import { SlidesPage } from "@/components/SlidesPage";

export const metadata: Metadata = { title: "Slides" };

export default function Page() {
  return <SlidesPage />;
}
