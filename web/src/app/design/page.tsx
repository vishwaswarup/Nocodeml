import type { Metadata } from "next";
import { DesignShowcase } from "./showcase";

export const metadata: Metadata = { title: "Design system · NoCodeML" };

export default function DesignPage() {
  return <DesignShowcase />;
}
