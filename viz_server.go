package main

import (
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"time"
)

const port = ":8001" // Different port from the tween viz_server

func main() {
	// Get the directory of the server
	vizDir, err := os.Getwd()
	if err != nil {
		log.Fatal(err)
	}

	// Data file path
	resultsPath := filepath.Join(vizDir, "arc_benchmark_results.json")

	// Check if results exist, if not run benchmark
	if _, err := os.Stat(resultsPath); os.IsNotExist(err) {
		fmt.Println("🚀 Results not found. Running ARC-AGI Benchmark...")
		fmt.Println("This compares 6 training modes across ARC-AGI data.")
		fmt.Println("May take a few minutes as it runs all modes in parallel.\n")

		cmd := exec.Command("go", "run", "arc_benchmark.go")
		cmd.Dir = vizDir
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		if err := cmd.Run(); err != nil {
			log.Fatalf("❌ Error running benchmark: %v", err)
		}

		if _, err := os.Stat(resultsPath); err == nil {
			fmt.Println("✅ Benchmark complete. Results saved to arc_benchmark_results.json")
		} else {
			log.Fatal("❌ Benchmark failed or arc_benchmark_results.json was not created.")
		}
	}

	// Serve static files
	fs := http.FileServer(http.Dir(vizDir))
	http.Handle("/", fs)

	fmt.Printf("🌐 Starting ARC-AGI visualization server at http://localhost%s\n", port)
	fmt.Println("📊 Dashboard is live! Press Ctrl+C to stop.")

	// Serve the JSON file explicitly
	http.HandleFunc("/arc_benchmark_results.json", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		http.ServeFile(w, r, resultsPath)
	})

	// Serve ensemble fusion results
	http.HandleFunc("/ensemble_fusion_arc_agi1.json", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		http.ServeFile(w, r, filepath.Join(vizDir, "test43a_ensemble_fusion_2", "ensemble_fusion_results_arc_agi1.json"))
	})

	http.HandleFunc("/ensemble_fusion_arc_agi2.json", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		http.ServeFile(w, r, filepath.Join(vizDir, "test43a_ensemble_fusion_2", "ensemble_fusion_results_2.json"))
	})

	// Open browser
	go func() {
		time.Sleep(1 * time.Second)
		openBrowser("http://localhost" + port)
	}()

	log.Fatal(http.ListenAndServe(port, nil))
}

func openBrowser(url string) {
	var err error
	switch runtime.GOOS {
	case "linux":
		err = exec.Command("xdg-open", url).Start()
	case "windows":
		err = exec.Command("rundll32", "url.dll,FileProtocolHandler", url).Start()
	case "darwin":
		err = exec.Command("open", url).Start()
	default:
		err = fmt.Errorf("unsupported platform")
	}
	if err != nil {
		fmt.Printf("Please open %s in your browser.\n", url)
	}
}
