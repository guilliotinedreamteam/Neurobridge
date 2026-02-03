import { SidebarProvider, SidebarTrigger, SidebarInset } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/AppSidebar";

export default function Layout({ children }: { children: React.ReactNode }) {
  return (
    <SidebarProvider>
      <AppSidebar />
      <SidebarInset>
        <header className="flex h-16 shrink-0 items-center gap-2 border-b px-4">
            <SidebarTrigger className="-ml-1" />
            <div className="w-[1px] h-4 bg-gray-200 dark:bg-gray-700 mx-2" />
            <span className="font-medium">Dashboard</span>
        </header>
        <main className="flex flex-1 flex-col gap-4 p-4">
            {children}
        </main>
      </SidebarInset>
    </SidebarProvider>
  );
}