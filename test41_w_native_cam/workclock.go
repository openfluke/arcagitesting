package main

import (
	"runtime"
	"syscall"
	"time"
)

// workSpan times infer/train on the current OS thread.
// Prefer Linux RUSAGE_THREAD CPU time so concurrent workers don't inflate
// duty-cycle with scheduler wait; fall back to wall clock otherwise.
type workSpan struct {
	cpu0  time.Duration
	wall0 time.Time
	cpuOK bool
}

func startWork() workSpan {
	c, ok := threadCPU()
	return workSpan{cpu0: c, wall0: time.Now(), cpuOK: ok}
}

func (w workSpan) elapsed() time.Duration {
	if w.cpuOK {
		c, ok := threadCPU()
		if ok && c >= w.cpu0 {
			return c - w.cpu0
		}
	}
	return time.Since(w.wall0)
}

func threadCPU() (time.Duration, bool) {
	if runtime.GOOS != "linux" {
		return 0, false
	}
	var ru syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_THREAD, &ru); err != nil {
		return 0, false
	}
	return timevalDuration(ru.Utime) + timevalDuration(ru.Stime), true
}

func timevalDuration(tv syscall.Timeval) time.Duration {
	return time.Duration(tv.Sec)*time.Second + time.Duration(tv.Usec)*time.Microsecond
}

func dutyClockName() string {
	if runtime.GOOS == "linux" {
		return "thread-CPU (RUSAGE_THREAD)"
	}
	return "wall-clock"
}
