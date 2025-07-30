'use client';

import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels';
import { Separator } from '@/components/ui/separator';
import ChatContainer from '@/components/chat/ChatContainer';
import ChatInput from '@/components/chat/ChatInput';
import PipelineSettings from '@/components/settings/PipelineSettings';
import { cn } from '@/lib/utils';

export default function HomePage() {
  const [showSettings, setShowSettings] = React.useState(false);

  const toggleSettings = () => setShowSettings(!showSettings);

  return (
    <div className="h-screen flex flex-col bg-background">
      <div className="flex-1 overflow-hidden">
        <PanelGroup direction="horizontal" className="h-full">
          {/* Main Chat Panel */}
          <Panel defaultSize={showSettings ? 65 : 100} minSize={50}>
            <div className="h-full flex flex-col">
              <div className="flex-1 overflow-hidden">
                <ChatContainer />
              </div>
              <ChatInput 
                onToggleSettings={toggleSettings}
                showSettings={showSettings}
              />
            </div>
          </Panel>

          {/* Settings Panel */}
          <AnimatePresence>
            {showSettings && (
              <>
                <PanelResizeHandle className="w-px bg-border hover:bg-border/80 transition-colors" />
                <Panel defaultSize={35} minSize={25} maxSize={50}>
                  <motion.div
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 20 }}
                    transition={{ duration: 0.2 }}
                    className="h-full overflow-hidden"
                  >
                    <div className="h-full flex flex-col">
                      {/* Settings Header */}
                      <div className="flex items-center justify-between p-4 border-b">
                        <h2 className="font-semibold">Pipeline Settings</h2>
                        <button
                          onClick={toggleSettings}
                          className="text-muted-foreground hover:text-foreground transition-colors text-xl leading-none"
                        >
                          ×
                        </button>
                      </div>

                      {/* Settings Content */}
                      <div className="flex-1 overflow-auto custom-scrollbar">
                        <div className="p-4">
                          <PipelineSettings />
                        </div>
                      </div>
                    </div>
                  </motion.div>
                </Panel>
              </>
            )}
          </AnimatePresence>
        </PanelGroup>
      </div>
    </div>
  );
}