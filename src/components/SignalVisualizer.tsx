import { useEffect, useRef, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export function SignalVisualizer() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const ws = new WebSocket("ws://localhost:8000/ws/signals");
    wsRef.current = ws;

    ws.onopen = () => setIsConnected(true);
    ws.onclose = () => setIsConnected(false);

    ws.onmessage = (event) => {
      const payload = JSON.parse(event.data);
      const channels = payload.channels as number[]; // array of 128 floats
      draw(channels);
    };

    return () => {
      ws.close();
    };
  }, []);

  const draw = (data: number[]) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    
    // Clear background
    ctx.fillStyle = "#111"; // Dark background
    ctx.fillRect(0, 0, width, height);

    // Draw channels
    // We display 128 channels as vertical bars or lines? 
    // Let's do a heatmap style or just a subset of lines.
    // 128 lines is too crowded. Let's draw top 16 channels as lines.
    
    const numChannelsToDraw = 16;
    const rowHeight = height / numChannelsToDraw;

    ctx.strokeStyle = "#00ff00";
    ctx.lineWidth = 2;
    ctx.beginPath();

    for (let i = 0; i < numChannelsToDraw; i++) {
        const val = data[i]; // roughly -1 to 1
        // Map val to y-coord centered in the row
        const cy = i * rowHeight + rowHeight / 2;
        // Draw a "dot" or "bar" representing current amplitude? 
        // A single frame isn't enough for a waveform history unless we buffer it.
        // For now, let's just visualize INSTANTANEOUS amplitude as bars.
        
        const barWidth = (val + 1) / 2 * width; // map -1..1 to 0..width
        
        // Draw bar
        ctx.fillStyle = `hsl(${i * 20}, 100%, 50%)`;
        ctx.fillRect(0, i * rowHeight + 2, barWidth, rowHeight - 4);
    }
  };

  return (
    <Card className="col-span-3">
      <CardHeader>
        <CardTitle className="flex justify-between">
            <span>Real-time Signals</span>
            <span className={isConnected ? "text-green-500" : "text-red-500"}>
                {isConnected ? "LIVE" : "OFFLINE"}
            </span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <canvas 
            ref={canvasRef} 
            width={800} 
            height={400} 
            className="w-full h-[300px] rounded border bg-black"
        />
      </CardContent>
    </Card>
  );
}