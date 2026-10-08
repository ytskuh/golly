// This file is part of Golly.
// See docs/License.html for the copyright notice.

#include "wx/wxprec.h"      // for compilers that support precompilation
#ifndef WX_PRECOMP
    #include "wx/wx.h"      // for all others include the necessary headers
#endif

#include <algorithm>        // for std::sort
#include <chrono>           // for std::chrono::steady_clock
#include <vector>

#include "wx/filename.h"   // for wxFileName

#include "bigint.h"
#include "lifealgo.h"

#include "wxgolly.h"       // for wxGetApp, mainptr, viewptr
#include "wxmain.h"        // for mainptr->...
#include "wxview.h"        // for viewptr->...
#include "wxalgos.h"       // for NumAlgos, GetAlgoName
#include "wxlayer.h"       // for currlayer
#include "wxbench.h"

// -----------------------------------------------------------------------------

bool benchmode = false;

// settings from the command line
static wxString patternpath;
static wxString outpath;
static wxString algoname;
static int basestep = 0;            // 0 means algorithm's default
static int stepexpo = 0;
static bigint targetgen = -1;       // < 0 means no generation limit
static double maxsecs = 0.0;        // <= 0 means no time limit
static int wantwd = 0, wantht = 0;  // requested viewport size (0 = leave alone)
static bool fitview = true;         // fit pattern in view before starting?
static int startmag = 0;            // used if fitview is false
static bool autofit = false;
static int benchthreads = 0;        // 0 means the par_threads default

// benchmark sequence
enum { SIZING, LOADING, WAITPAINT, RUNNING, DONE };
static int state = SIZING;
static int sizetries = 0;
static int paintsatload = 0;

// measurements
static bool running = false;
static double t0, t1;
static bigint startgen, endgen;
static wxString stopreason;
static double steptime = 0.0, yieldtime = 0.0, drawtime = 0.0, swaptime = 0.0, statustime = 0.0;
static int paints = 0;
static int frames = 0;
static int batches = 0;
static std::vector<double> stepms;     // time of each step (StepPattern call)
static std::vector<double> framems;    // draw + swap time of each viewport paint
static double stepbegin, yieldatbegin;
static double batchbegin, yieldatbatch;
static double batchtime = 0.0;

// -----------------------------------------------------------------------------

double BenchNow()
{
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// -----------------------------------------------------------------------------

static void BenchUsage(const wxString& msg)
{
    wxPrintf(wxT("golly --bench: %s\n"), msg);
    wxPrintf(wxT("usage: golly --bench pattern=FILE out=FILE [algo=NAME] [base=N] [expo=N]\n"
                 "       [gen=N] [secs=S] [size=WxH] [mag=fit|N] [autofit=0|1] [threads=N]\n"));
    exit(2);
}

bool BenchParseArgs(int argc, wxChar** argv, const wxString& initdir)
{
    if (argc < 2 || wxString(argv[1]) != wxT("--bench")) return false;
    benchmode = true;

    for (int n = 2; n < argc; n++) {
        wxString arg(argv[n]);
        wxString key = arg.BeforeFirst('=');
        wxString value = arg.AfterFirst('=');
        long i;
        if (key == wxT("pattern")) {
            patternpath = value;
        } else if (key == wxT("out")) {
            outpath = value;
        } else if (key == wxT("algo")) {
            algoname = value;
        } else if (key == wxT("base") && value.ToLong(&i) && i >= 2) {
            basestep = (int)i;
        } else if (key == wxT("expo") && value.ToLong(&i) && i >= 0) {
            stepexpo = (int)i;
        } else if (key == wxT("gen") && !value.IsEmpty()) {
            targetgen = bigint(value.mb_str(wxConvLocal));
        } else if (key == wxT("secs") && value.ToDouble(&maxsecs)) {
        } else if (key == wxT("size")) {
            long w, h;
            if (!value.BeforeFirst('x').ToLong(&w) || !value.AfterFirst('x').ToLong(&h))
                BenchUsage(wxT("bad size: ") + value);
            wantwd = (int)w;
            wantht = (int)h;
        } else if (key == wxT("mag")) {
            if (value == wxT("fit")) {
                fitview = true;
            } else if (value.ToLong(&i)) {
                fitview = false;
                startmag = (int)i;
            } else {
                BenchUsage(wxT("bad mag: ") + value);
            }
        } else if (key == wxT("autofit") && value.ToLong(&i)) {
            autofit = i != 0;
        } else if (key == wxT("threads") && value.ToLong(&i) && i >= 1) {
            benchthreads = (int)i;
        } else {
            BenchUsage(wxT("bad argument: ") + arg);
        }
    }

    if (patternpath.IsEmpty()) BenchUsage(wxT("pattern=FILE is required"));
    if (outpath.IsEmpty()) BenchUsage(wxT("out=FILE is required"));
    wxFileName pfile(patternpath), ofile(outpath);
    pfile.MakeAbsolute(initdir);
    ofile.MakeAbsolute(initdir);
    patternpath = pfile.GetFullPath();
    outpath = ofile.GetFullPath();
    if (targetgen < 0 && maxsecs <= 0.0) BenchUsage(wxT("gen=N or secs=S is required"));
    return true;
}

// -----------------------------------------------------------------------------

static void JsonString(FILE* f, const char* key, const wxString& s)
{
    wxString esc = s;
    esc.Replace(wxT("\\"), wxT("\\\\"));
    esc.Replace(wxT("\""), wxT("\\\""));
    fprintf(f, "  \"%s\": \"%s\",\n", key, (const char*)esc.mb_str(wxConvUTF8));
}

static void JsonPercentiles(FILE* f, const char* key, std::vector<double> v)
{
    // v is a copy so we can sort it
    std::sort(v.begin(), v.end());
    if (v.empty()) {
        fprintf(f, "  \"%s\": null,\n", key);
        return;
    }
    double sum = 0.0;
    for (double x : v) sum += x;
    size_t n = v.size();
    fprintf(f, "  \"%s\": {\"mean\": %.4f, \"p50\": %.4f, \"p95\": %.4f, \"max\": %.4f},\n", key,
            sum / n, v[n / 2], v[std::min(n - 1, (size_t)(n * 0.95))], v[n - 1]);
}

static void JsonArray(FILE* f, const char* key, const std::vector<double>& v, bool last)
{
    fprintf(f, "  \"%s\": [", key);
    for (size_t i = 0; i < v.size(); i++) fprintf(f, i ? ", %.4f" : "%.4f", v[i]);
    fprintf(f, last ? "]\n" : "],\n");
}

static void BenchFinish()
{
    state = DONE;
    lifealgo* algo = currlayer->algo;

    FILE* f = fopen(outpath.mb_str(wxConvLocal), "w");
    if (f == NULL) {
        wxPrintf(wxT("golly --bench: could not write %s\n"), outpath);
        exit(1);
    }

    double wall = t1 - t0;
    double gens = endgen.todouble() - startgen.todouble();
    fprintf(f, "{\n");
    JsonString(f, "pattern", patternpath);
    JsonString(f, "algo", wxString(GetAlgoName(currlayer->algtype), wxConvLocal));
    JsonString(f, "rule", wxString(algo->getrule(), wxConvLocal));
    JsonString(f, "stop_reason", stopreason);
    fprintf(f, "  \"base\": %d,\n  \"expo\": %d,\n", currlayer->currbase, currlayer->currexpo);
    fprintf(f, "  \"increment\": \"%s\",\n", algo->getIncrement().tostring(0));
    fprintf(f, "  \"view_wd\": %d,\n  \"view_ht\": %d,\n",
            currlayer->view->getwidth(), currlayer->view->getheight());
    fprintf(f, "  \"mag\": %d,\n", viewptr->GetMag());
    fprintf(f, "  \"autofit\": %s,\n", autofit ? "true" : "false");
    fprintf(f, "  \"par_threads\": %d,\n", parthreads);
    fprintf(f, "  \"start_gen\": \"%s\",\n", startgen.tostring(0));
    fprintf(f, "  \"end_gen\": \"%s\",\n", endgen.tostring(0));
    fprintf(f, "  \"pop\": \"%s\",\n", algo->getPopulation().tostring(0));
    if (algo->isEmpty()) {
        fprintf(f, "  \"bbox\": null,\n");
    } else {
        bigint t, l, b, r;
        algo->findedges(&t, &l, &b, &r);
        fprintf(f, "  \"bbox\": [\"%s\",", l.tostring(0));
        fprintf(f, " \"%s\",", t.tostring(0));
        fprintf(f, " \"%s\",", r.tostring(0));
        fprintf(f, " \"%s\"],\n", b.tostring(0));
    }
    fprintf(f, "  \"gens\": %.17g,\n", gens);
    fprintf(f, "  \"wall_s\": %.6f,\n", wall);
    fprintf(f, "  \"gens_per_s\": %.6g,\n", wall > 0.0 ? gens / wall : 0.0);
    int steps = (int)stepms.size();
    fprintf(f, "  \"steps\": %d,\n", steps);
    fprintf(f, "  \"steps_per_s\": %.6g,\n", wall > 0.0 ? steps / wall : 0.0);
    fprintf(f, "  \"step_only_steps_per_s\": %.6g,\n", steptime > 0.0 ? steps / steptime : 0.0);
    fprintf(f, "  \"batches\": %d,\n", batches);
    fprintf(f, "  \"frames\": %d,\n", frames);
    fprintf(f, "  \"batch_s\": %.6f,\n", batchtime);
    fprintf(f, "  \"step_s\": %.6f,\n", steptime);
    fprintf(f, "  \"draw_s\": %.6f,\n", drawtime);
    fprintf(f, "  \"swap_s\": %.6f,\n", swaptime);
    fprintf(f, "  \"status_s\": %.6f,\n", statustime);
    fprintf(f, "  \"yield_s\": %.6f,\n", yieldtime);
    fprintf(f, "  \"other_s\": %.6f,\n", wall - batchtime - drawtime - swaptime - statustime);
    JsonPercentiles(f, "step_ms", stepms);
    JsonPercentiles(f, "frame_ms", framems);
    JsonArray(f, "step_ms_all", stepms, false);
    JsonArray(f, "frame_ms_all", framems, true);
    fprintf(f, "}\n");
    fclose(f);

    // true means the close can't be vetoed, so no "save changes" dialog
    mainptr->Close(true);
}

// -----------------------------------------------------------------------------

static void BenchBeginRun()
{
    steptime = yieldtime = drawtime = swaptime = statustime = batchtime = 0.0;
    frames = batches = 0;
    stepms.clear();
    framems.clear();
    startgen = currlayer->algo->getGeneration();
    state = RUNNING;
    running = true;
    t0 = BenchNow();
    mainptr->StartGenerating();
    if (!mainptr->generating) {
        // eg. pattern is empty
        t1 = BenchNow();
        running = false;
        endgen = currlayer->algo->getGeneration();
        stopreason = wxT("not_started");
        BenchFinish();
    }
}

class BenchTimer : public wxTimer
{
public:
    virtual void Notify();
};

static BenchTimer* benchtimerobj = NULL;

void BenchTimer::Notify()
{
    if (state == SIZING) {
        // resize the main window until the viewport has the requested size
        int wd, ht;
        viewptr->GetClientSize(&wd, &ht);
        if (wantwd > 0 && (wd != wantwd || ht != wantht) && sizetries < 40) {
            wxSize sz = mainptr->GetSize();
            mainptr->SetSize(sz.x + wantwd - wd, sz.y + wantht - ht);
            sizetries++;
            return;
        }
        state = LOADING;
    } else if (state == LOADING) {
        if (benchthreads > 0) SetParallelThreads(benchthreads);
        if (!algoname.IsEmpty()) {
            int i;
            for (i = 0; i < NumAlgos(); i++) {
                if (algoname == wxString(GetAlgoName(i), wxConvLocal)) break;
            }
            if (i == NumAlgos()) {
                wxPrintf(wxT("golly --bench: unknown algo %s\n"), algoname);
                exit(2);
            }
            // LoadPattern tries the current algorithm first
            if (i != currlayer->algtype) mainptr->ChangeAlgorithm(i);
        }
        mainptr->OpenFile(patternpath, false);
        if (basestep > 0) currlayer->currbase = basestep;
        mainptr->SetStepExponent(stepexpo);
        currlayer->autofit = autofit;
        if (fitview) {
            viewptr->FitPattern();
        } else {
            viewptr->SetMag(startmag);
        }
        mainptr->UpdateEverything();
        paintsatload = paints;
        state = WAITPAINT;
    } else if (state == WAITPAINT) {
        if (paints > paintsatload) {
            Stop();
            BenchBeginRun();
        }
    }
}

void BenchStart()
{
    benchtimerobj = new BenchTimer();
    benchtimerobj->Start(50, wxTIMER_CONTINUOUS);
}

// -----------------------------------------------------------------------------

void BenchStepBegin()
{
    stepbegin = BenchNow();
    yieldatbegin = yieldtime;
}

void BenchStepEnd()
{
    if (!running) return;
    double secs = BenchNow() - stepbegin - (yieldtime - yieldatbegin);
    steptime += secs;
    stepms.push_back(secs * 1000.0);
}

void BenchBatchBegin()
{
    batchbegin = BenchNow();
    yieldatbatch = yieldtime;
}

void BenchBatchEnd()
{
    // the run's last batch was counted by BenchStopNow
    if (!running) return;
    batchtime += BenchNow() - batchbegin - (yieldtime - yieldatbatch);
    batches++;
}

void BenchAddYield(double secs)
{
    if (running) yieldtime += secs;
}

void BenchAddPaint(double drawsecs, double swapsecs)
{
    paints++;
    if (!running) return;
    drawtime += drawsecs;
    swaptime += swapsecs;
    frames++;
    framems.push_back((drawsecs + swapsecs) * 1000.0);
}

void BenchAddStatus(double secs)
{
    if (running) statustime += secs;
}

bool BenchStopNow()
{
    if (!running) return false;
    double now = BenchNow();
    bigint gen = currlayer->algo->getGeneration();
    if (targetgen >= 0 && gen >= targetgen) {
        stopreason = wxT("gen");
    } else if (maxsecs > 0.0 && now - t0 >= maxsecs) {
        stopreason = wxT("secs");
    } else {
        return false;
    }
    t1 = now;
    endgen = gen;
    // count this batch's time up to t1 (BenchBatchEnd ignores it)
    batchtime += now - batchbegin - (yieldtime - yieldatbatch);
    batches++;
    running = false;
    // write results once generating has stopped and OnGenTimer has returned
    wxGetApp().CallAfter([] { BenchFinish(); });
    return true;
}
