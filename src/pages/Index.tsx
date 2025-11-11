import { MadeWithGiga } from "@/components/made-with-giga";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Brain, Zap, Activity, Github, BookOpen, Code } from "lucide-react";

const Index = () => {
  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800">
      {/* Hero Section */}
      <div className="container mx-auto px-4 py-16">
        <div className="text-center mb-16">
          <div className="flex justify-center mb-6">
            <div className="p-4 bg-blue-100 dark:bg-blue-900 rounded-full">
              <Brain className="w-16 h-16 text-blue-600 dark:text-blue-400" />
            </div>
          </div>
          <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
            NeuroBridge
          </h1>
          <p className="text-xl text-gray-600 dark:text-gray-300 max-w-2xl mx-auto mb-8">
            A neural interface bridge system for processing and analyzing brain-computer interface data
          </p>
          <div className="flex gap-4 justify-center flex-wrap">
            <Button size="lg" className="gap-2">
              <Github className="w-5 h-5" />
              View on GitHub
            </Button>
            <Button size="lg" variant="outline" className="gap-2">
              <BookOpen className="w-5 h-5" />
              Documentation
            </Button>
          </div>
        </div>

        {/* Features Grid */}
        <div className="grid md:grid-cols-3 gap-6 mb-16">
          <Card className="hover:shadow-lg transition-shadow">
            <CardHeader>
              <div className="w-12 h-12 bg-blue-100 dark:bg-blue-900 rounded-lg flex items-center justify-center mb-4">
                <Zap className="w-6 h-6 text-blue-600 dark:text-blue-400" />
              </div>
              <CardTitle>Real-time Processing</CardTitle>
              <CardDescription>
                Process neural signals in real-time with low latency and high accuracy
              </CardDescription>
            </CardHeader>
          </Card>

          <Card className="hover:shadow-lg transition-shadow">
            <CardHeader>
              <div className="w-12 h-12 bg-purple-100 dark:bg-purple-900 rounded-lg flex items-center justify-center mb-4">
                <Activity className="w-6 h-6 text-purple-600 dark:text-purple-400" />
              </div>
              <CardTitle>Advanced Analysis</CardTitle>
              <CardDescription>
                Sophisticated algorithms for signal filtering and pattern recognition
              </CardDescription>
            </CardHeader>
          </Card>

          <Card className="hover:shadow-lg transition-shadow">
            <CardHeader>
              <div className="w-12 h-12 bg-green-100 dark:bg-green-900 rounded-lg flex items-center justify-center mb-4">
                <Code className="w-6 h-6 text-green-600 dark:text-green-400" />
              </div>
              <CardTitle>Modular Design</CardTitle>
              <CardDescription>
                Extensible architecture that adapts to your specific needs
              </CardDescription>
            </CardHeader>
          </Card>
        </div>

        {/* Quick Start Section */}
        <Card className="mb-16">
          <CardHeader>
            <CardTitle className="text-2xl">Quick Start</CardTitle>
            <CardDescription>Get up and running in minutes</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="bg-gray-900 text-gray-100 p-4 rounded-lg font-mono text-sm overflow-x-auto">
              <div className="mb-2"># Clone the repository</div>
              <div className="text-green-400">git clone https://github.com/yourusername/neurobridge.git</div>
              <div className="mt-4 mb-2"># Install Python dependencies</div>
              <div className="text-green-400">pip install -r requirements.txt</div>
              <div className="mt-4 mb-2"># Install Node.js dependencies</div>
              <div className="text-green-400">npm install</div>
              <div className="mt-4 mb-2"># Start the development server</div>
              <div className="text-green-400">npm run dev</div>
            </div>
          </CardContent>
        </Card>

        {/* Tech Stack */}
        <div className="text-center mb-16">
          <h2 className="text-3xl font-bold mb-8">Built With Modern Technologies</h2>
          <div className="flex flex-wrap justify-center gap-4">
            {["Python", "React", "TypeScript", "NumPy", "SciPy", "Tailwind CSS"].map((tech) => (
              <div
                key={tech}
                className="px-6 py-3 bg-white dark:bg-gray-800 rounded-full shadow-md hover:shadow-lg transition-shadow"
              >
                {tech}
              </div>
            ))}
          </div>
        </div>

        {/* CTA Section */}
        <div className="text-center bg-gradient-to-r from-blue-600 to-purple-600 rounded-2xl p-12 text-white">
          <h2 className="text-3xl font-bold mb-4">Ready to Get Started?</h2>
          <p className="text-xl mb-8 opacity-90">
            Join the community and start building neural interfaces today
          </p>
          <Button size="lg" variant="secondary" className="gap-2">
            <Github className="w-5 h-5" />
            Star on GitHub
          </Button>
        </div>
      </div>

      <MadeWithGiga />
    </div>
  );
};

export default Index;