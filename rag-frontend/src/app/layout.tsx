import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import { Toaster } from '@/components/ui/toaster';
import './globals.css';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'RAG Pipeline',
  description: 'A modern interface for RAG pipeline management and querying',
  keywords: ['RAG', 'AI', 'Machine Learning', 'Retrieval', 'Pipeline'],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.className} antialiased`}>
        <div className="min-h-screen bg-background text-foreground">
          {children}
        </div>
        <Toaster />
      </body>
    </html>
  );
}