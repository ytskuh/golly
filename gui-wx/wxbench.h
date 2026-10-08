// This file is part of Golly.
// See docs/License.html for the copyright notice.

#ifndef _WXBENCH_H_
#define _WXBENCH_H_

// Benchmark mode.  Starting Golly with
//
//    golly --bench pattern=FILE out=FILE [key=value ...]
//
// loads the pattern, generates it with the normal GUI loop (OnGenTimer,
// StepPattern, OnPaint) until a target generation or time limit is reached,
// writes timing results as JSON to the out file, then quits.
// See bench/README.md for the list of keys.

extern bool benchmode;      // true if --bench was given on the command line

bool BenchParseArgs(int argc, wxChar** argv, const wxString& initdir);
// Return true if argv[1] is "--bench", in which case the remaining arguments
// are parsed as key=value settings (and should not be treated as files).
// Relative file paths are resolved against initdir.

void BenchStart();
// Called once the main window is shown; starts the benchmark sequence.

double BenchNow();
// Return a monotonic time stamp in seconds.

// Timing hooks.  They only record data while a benchmark run is in progress.

void BenchStepBegin();
void BenchStepEnd();
// Called around the step code in StepPattern.  Time spent in Yield calls
// made by the poller during the step is excluded.

void BenchBatchBegin();
void BenchBatchEnd();
// Called around the body of OnGenTimer (one StepPattern call, which
// also updates the display).  Time spent in Yield calls is excluded.

void BenchAddYield(double secs);
// Called by the poller after each Yield.

void BenchAddPaint(double drawsecs, double swapsecs);
// Called by PatternView::OnPaint (DrawView time, SwapBuffers time).

void BenchAddStatus(double secs);
// Called by StatusBar::OnPaint.

bool BenchStopNow();
// Called at the end of each successful StepPattern; returns true if the
// benchmark run is finished and generating should stop.

#endif
