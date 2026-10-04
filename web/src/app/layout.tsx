import type { Metadata } from "next";
import { DM_Mono, Geist } from "next/font/google";
import { MotionProvider } from "@/components/motion-provider";
import "./globals.css";

const geist = Geist({ variable: "--font-geist", subsets: ["latin"] });
const dmMono = DM_Mono({ variable: "--font-dm-mono", subsets: ["latin"], weight: ["400", "500"] });

export const metadata: Metadata = {
  title: "NoCodeML",
  description: "Build correct, reproducible, explainable machine-learning pipelines without writing code.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${geist.variable} ${dmMono.variable} h-full`}>
      <body className="min-h-full">
        <MotionProvider>{children}</MotionProvider>
      </body>
    </html>
  );
}
