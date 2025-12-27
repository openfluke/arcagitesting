package main

import (
	"fmt"
	"net/http"
	"os"
)

func main() {
	port := "8043"
	if len(os.Args) > 1 {
		port = os.Args[1]
	}

	// Serve static files from current directory
	fs := http.FileServer(http.Dir("."))
	http.Handle("/", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Set correct MIME types
		switch {
		case r.URL.Path == "/" || r.URL.Path == "/index.html":
			w.Header().Set("Content-Type", "text/html")
		case r.URL.Path == "/main.wasm":
			w.Header().Set("Content-Type", "application/wasm")
		}
		fs.ServeHTTP(w, r)
	}))

	fmt.Println("🌐 ════════════════════════════════════════════════════════════════")
	fmt.Printf("🔮 ARC-AGI Browser Solver running at http://localhost:%s\n", port)
	fmt.Println("🎮 Open in your browser to solve ARC tasks with EXPLOSIONS! 💥")
	fmt.Println("🌐 ════════════════════════════════════════════════════════════════")
	http.ListenAndServe(":"+port, nil)
}
