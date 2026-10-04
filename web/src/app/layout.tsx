import type { Metadata } from "next";
import { DM_Mono, Geist } from "next/font/google";
import { AuthProvider } from "@/components/auth-provider";
import { MotionProvider } from "@/components/motion-provider";
import "./globals.css";

const geist = Geist({ variable: "--font-geist", subsets: ["latin"] });
const dmMono = DM_Mono({ variable: "--font-dm-mono", subsets: ["latin"], weight: ["400", "500"] });

export const metadata: Metadata = {
  title: "NoCodeML",
  description: "Run, compare and tweak machine-learning experiments in your browser, without rewriting Python for every test.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${geist.variable} ${dmMono.variable} h-full`}>
      <body className="min-h-full">
        <MotionProvider>
          <AuthProvider>{children}</AuthProvider>
        </MotionProvider>
      </body>
    </html>
  );
}
