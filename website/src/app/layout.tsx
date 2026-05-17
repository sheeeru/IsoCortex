import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Toaster } from "@/components/ui/toaster";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "IsoCortex — High-Performance Local Neural Information Retrieval Engine",
  description:
    "100% local semantic search engine. Index 20+ file formats, search millions of documents with sub-millisecond latency using HNSW graphs and neural embeddings. Zero cloud dependency.",
  keywords: [
    "IsoCortex",
    "semantic search",
    "neural information retrieval",
    "HNSW",
    "vector search",
    "local search",
    "privacy",
    "GDPR",
    "offline search",
    "embeddings",
    "RAG",
    "document search",
  ],
  authors: [{ name: "Shaheer Qureshi" }],
  icons: {
    icon: "/favicon.png",
  },
  openGraph: {
    title: "IsoCortex — Local Neural Information Retrieval Engine",
    description:
      "Sub-millisecond semantic search across 20+ file formats. 100% local, zero cloud dependency.",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "IsoCortex — Local Neural Search",
    description:
      "Sub-millisecond semantic search. 100% local. Zero cloud.",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-background text-foreground`}
      >
        {children}
        <Toaster />
      </body>
    </html>
  );
}
