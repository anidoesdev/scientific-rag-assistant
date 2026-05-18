import "./globals.css";
import type { Metadata } from "next";
import { Inter, Lora } from "next/font/google";
import Script from "next/script";
import { AuthProvider } from "./contexts/AuthContext";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });
const lora  = Lora({ subsets: ["latin"], variable: "--font-lora" });

export const metadata: Metadata = {
  title: "Scientific RAG Assistant",
  description: "Minimal scientific paper Q&A assistant"
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${inter.variable} ${lora.variable}`}>
      <body className={inter.className}>
        <AuthProvider>
          {children}
        </AuthProvider>
        <Script
          src="https://accounts.google.com/gsi/client"
          strategy="afterInteractive"
        />
      </body>
    </html>
  );
}
