import { useEffect, useMemo, useRef, useState } from "react";
import { Play, Square, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";

type EventType =
  | "thought"
  | "action"
  | "observation"
  | "final"
  | "error"
  | "info";

interface TraceEvent {
  type: EventType;
  content: string;
  ts: number;
}

interface Template {
  id: string;
  label: string;
  description: string;
  tools: string[];
  example_task: string;
}

const FALLBACK_TEMPLATES: Template[] = [
  {
    id: "math_solver",
    label: "Math Solver",
    description: "ReAct agent with the calculator tool.",
    tools: ["calculator", "datetime_tool"],
    example_task: "What is 2 ** 16 divided by 4?",
  },
  {
    id: "researcher",
    label: "Researcher",
    description: "Search then summarise.",
    tools: ["web_search_mock", "text_summarizer"],
    example_task: "Give me a short summary of agentic AI systems.",
  },
  {
    id: "assistant",
    label: "General Assistant",
    description: "All built-in tools available.",
    tools: ["calculator", "datetime_tool", "web_search_mock", "text_summarizer"],
    example_task: "What time is it right now?",
  },
];

const ROLE_STYLES: Record<EventType, string> = {
  thought: "text-sky-400",
  action: "text-amber-400",
  observation: "text-emerald-400",
  final: "text-fuchsia-400 font-semibold",
  error: "text-red-400",
  info: "text-muted-foreground",
};

const ROLE_LABELS: Record<EventType, string> = {
  thought: "Thought",
  action: "Action",
  observation: "Observation",
  final: "Final Answer",
  error: "Error",
  info: "Info",
};

function backendHttpUrl(): string {
  const ws = import.meta.env.VITE_BACKEND_WS_URL ?? "ws://localhost:8000/ws/run";
  return ws.replace(/^ws/, "http").replace(/\/ws\/run$/, "");
}

export default function App() {
  const [templates, setTemplates] = useState<Template[]>(FALLBACK_TEMPLATES);
  const [templateId, setTemplateId] = useState<string>(FALLBACK_TEMPLATES[0].id);
  const [task, setTask] = useState<string>(FALLBACK_TEMPLATES[0].example_task);
  const [events, setEvents] = useState<TraceEvent[]>([]);
  const [running, setRunning] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const traceRef = useRef<HTMLDivElement | null>(null);

  const selectedTemplate = useMemo(
    () => templates.find((t) => t.id === templateId) ?? templates[0],
    [templates, templateId],
  );

  useEffect(() => {
    fetch(`${backendHttpUrl()}/templates`)
      .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
      .then((data) => {
        if (Array.isArray(data?.templates) && data.templates.length > 0) {
          setTemplates(data.templates);
          setTemplateId(data.templates[0].id);
          setTask(data.templates[0].example_task);
        }
      })
      .catch(() => {
        // Fall back to the hard-coded list — backend may not be up yet.
      });
  }, []);

  useEffect(() => {
    if (traceRef.current) {
      traceRef.current.scrollTop = traceRef.current.scrollHeight;
    }
  }, [events]);

  const startRun = () => {
    if (running || !task.trim()) return;
    setEvents([]);
    setRunning(true);

    const url =
      import.meta.env.VITE_BACKEND_WS_URL ?? "ws://localhost:8000/ws/run";
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      ws.send(JSON.stringify({ template: templateId, task }));
    };
    ws.onmessage = (msg) => {
      try {
        const evt: TraceEvent = JSON.parse(msg.data);
        setEvents((prev) => [...prev, evt]);
      } catch {
        // ignore malformed frames
      }
    };
    ws.onerror = () => {
      setEvents((prev) => [
        ...prev,
        {
          type: "error",
          content: "WebSocket error. Is the backend running?",
          ts: Date.now() / 1000,
        },
      ]);
    };
    ws.onclose = () => {
      setRunning(false);
      wsRef.current = null;
    };
  };

  const stopRun = () => {
    wsRef.current?.close();
    setRunning(false);
  };

  return (
    <div className="min-h-screen bg-background text-foreground">
      <header className="border-b border-border px-6 py-4 flex items-center gap-2">
        <Sparkles className="h-5 w-5 text-primary" />
        <h1 className="text-lg font-semibold">Agentic Systems Playground</h1>
        <span className="ml-auto text-xs text-muted-foreground">
          ReAct trace, streamed live over WebSocket
        </span>
      </header>

      <main className="grid grid-cols-1 md:grid-cols-[380px_1fr] gap-4 p-4 md:p-6">
        {/* Left: config */}
        <Card>
          <CardHeader>
            <CardTitle>Configure run</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <label className="block text-xs font-medium mb-2 text-muted-foreground">
                Agent template
              </label>
              <div className="flex flex-col gap-2">
                {templates.map((t) => (
                  <button
                    key={t.id}
                    onClick={() => {
                      setTemplateId(t.id);
                      setTask(t.example_task);
                    }}
                    className={cn(
                      "rounded-md border px-3 py-2 text-left text-sm transition-colors",
                      templateId === t.id
                        ? "border-primary bg-primary/10"
                        : "border-border hover:bg-muted",
                    )}
                  >
                    <div className="font-medium">{t.label}</div>
                    <div className="text-xs text-muted-foreground">
                      {t.description}
                    </div>
                    <div className="text-[10px] mt-1 text-muted-foreground/80 font-mono">
                      tools: {t.tools.join(", ")}
                    </div>
                  </button>
                ))}
              </div>
            </div>

            <div>
              <label className="block text-xs font-medium mb-2 text-muted-foreground">
                Task
              </label>
              <textarea
                value={task}
                onChange={(e) => setTask(e.target.value)}
                rows={5}
                className="w-full rounded-md border border-border bg-background p-2 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-primary"
                placeholder="Ask the agent to do something..."
              />
            </div>

            <div className="flex gap-2">
              {!running ? (
                <Button onClick={startRun} disabled={!task.trim()}>
                  <Play className="h-4 w-4" /> Run
                </Button>
              ) : (
                <Button variant="outline" onClick={stopRun}>
                  <Square className="h-4 w-4" /> Stop
                </Button>
              )}
              <Button
                variant="ghost"
                onClick={() => setEvents([])}
                disabled={events.length === 0}
              >
                Clear trace
              </Button>
            </div>

            {selectedTemplate && (
              <p className="text-xs text-muted-foreground">
                Selected: <span className="font-mono">{selectedTemplate.label}</span>
              </p>
            )}
          </CardContent>
        </Card>

        {/* Right: streaming trace */}
        <Card className="min-h-[70vh] flex flex-col">
          <CardHeader className="border-b border-border">
            <CardTitle className="flex items-center gap-2">
              ReAct trace
              {running && (
                <span className="inline-block h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
              )}
            </CardTitle>
          </CardHeader>
          <CardContent className="flex-1 p-0">
            <div
              ref={traceRef}
              className="h-full max-h-[70vh] overflow-y-auto p-4 font-mono text-sm space-y-3"
            >
              {events.length === 0 && !running && (
                <p className="text-muted-foreground text-xs">
                  Press <span className="font-semibold">Run</span> to stream a ReAct
                  trace from the backend.
                </p>
              )}
              {events.map((e, i) => (
                <div key={i} className="leading-relaxed">
                  <span className={cn("mr-2", ROLE_STYLES[e.type])}>
                    [{ROLE_LABELS[e.type]}]
                  </span>
                  <span className="whitespace-pre-wrap">{e.content}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </main>

      <footer className="px-6 py-3 text-xs text-muted-foreground border-t border-border">
        Backend:{" "}
        <span className="font-mono">
          {import.meta.env.VITE_BACKEND_WS_URL ?? "ws://localhost:8000/ws/run"}
        </span>
      </footer>
    </div>
  );
}
