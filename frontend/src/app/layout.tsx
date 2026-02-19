import type { Metadata } from "next";
import { ThemeProvider } from "next-themes";
import "./globals.css";

export const metadata: Metadata = {
  title: "Strategy Governance Platform",
  description: "Strategy survivability, failure detection, and capital allocation governance",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>
        <ThemeProvider attribute="class" defaultTheme="dark" enableSystem>
          <div className="min-h-screen bg-gray-50 dark:bg-surface">
            <nav className="border-b border-gray-200 dark:border-gray-800">
              <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex items-center justify-between h-16">
                  <div className="flex items-center gap-8">
                    <a
                      href="/"
                      className="text-accent font-bold text-lg"
                      aria-label="Home"
                    >
                      Strategy Gov
                    </a>
                    <div className="hidden md:flex gap-6">
                      <a
                        href="/"
                        className="text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white text-sm"
                      >
                        Dashboard
                      </a>
                      <a
                        href="/early-warning"
                        className="text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white text-sm"
                      >
                        Early Warning
                      </a>
                      <a
                        href="/market"
                        className="text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white text-sm"
                      >
                        Market Regime
                      </a>
                      <a
                        href="/exposure"
                        className="text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white text-sm"
                      >
                        Exposure
                      </a>
                      <a
                        href="/candidates"
                        className="text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white text-sm"
                      >
                        Candidates
                      </a>
                      <a
                        href="/settings"
                        className="text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white text-sm"
                      >
                        Settings
                      </a>
                    </div>
                  </div>
                  <div className="flex items-center gap-4">
                    <span className="text-xs text-gray-400 dark:text-gray-500">
                      Can I still trust this strategy?
                    </span>
                  </div>
                </div>
              </div>
            </nav>
            <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
              {children}
            </main>
          </div>
        </ThemeProvider>
      </body>
    </html>
  );
}
