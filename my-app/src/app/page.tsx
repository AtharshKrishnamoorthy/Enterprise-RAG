import { ChatInterface } from '@/components/chat-interface';

export default function Home() {
  return (
    <main className="flex flex-col items-center justify-center min-h-screen bg-gray-100 dark:bg-gray-950 p-4">
      <div className="w-full max-w-3xl h-[80vh]">
        <ChatInterface />
      </div>
    </main>
  );
}





