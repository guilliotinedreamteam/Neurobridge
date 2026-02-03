import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Brain, Zap, Activity } from "lucide-react";
import { SignalVisualizer } from "@/components/SignalVisualizer";

const Index = () => {
  return (
    <div className="space-y-4">
        {/* Stats Grid */}
        <div className="grid gap-4 md:grid-cols-3">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">System Status</CardTitle>
              <Activity className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-green-600">Active</div>
              <p className="text-xs text-muted-foreground">Backend connected</p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Model Accuracy</CardTitle>
              <Brain className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">94.2%</div>
              <p className="text-xs text-muted-foreground">+2.1% from last run</p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Processing Latency</CardTitle>
              <Zap className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">12ms</div>
              <p className="text-xs text-muted-foreground">Real-time ready</p>
            </CardContent>
          </Card>
        </div>

        {/* Signal Visualizer */}
        <SignalVisualizer />

        {/* Control Panel Section */}
        <Card className="col-span-3">
          <CardHeader>
            <CardTitle>Control Panel</CardTitle>
            <CardDescription>Manage your NeuroBridge instance</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="p-4 border rounded-lg">
                <h3 className="font-semibold mb-2">Training</h3>
                <Button 
                  onClick={async () => {
                    try {
                      await import("@/lib/api").then(m => m.api.train());
                      alert("Training started!");
                    } catch (e) {
                      alert("Failed to start training");
                    }
                  }}
                  className="w-full bg-blue-600 hover:bg-blue-700"
                >
                  Start Training (10 Epochs)
                </Button>
              </div>
              
              <div className="p-4 border rounded-lg">
                <h3 className="font-semibold mb-2">Synthesis</h3>
                <div className="flex gap-2">
                  <input 
                    type="text" 
                    placeholder="Phoneme sequence (e.g. 0,5,12)" 
                    className="flex-1 p-2 border rounded"
                    id="synth-input"
                  />
                  <Button 
                    onClick={async () => {
                      const input = document.getElementById("synth-input") as HTMLInputElement;
                      if (!input.value) return;
                      try {
                        await import("@/lib/api").then(m => m.api.synthesize(input.value));
                        alert("Synthesis complete!");
                      } catch (e) {
                        alert("Synthesis failed");
                      }
                    }}
                    variant="secondary"
                  >
                    Synthesize
                  </Button>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
    </div>
  );
};

export default Index;