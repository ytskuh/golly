                        /*** /

This file is part of Golly, a Game of Life Simulator.
Copyright (C) 2011 Andrew Trevorrow and Tomas Rokicki.

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

 Web site:  http://sourceforge.net/projects/golly
 Authors:   rokicki@gmail.com  andrew@trevorrow.com

                        / ***/

#include "wx/wxprec.h"     // for compilers that support precompilation
#ifndef WX_PRECOMP
   #include "wx/wx.h"      // for all others include the necessary headers
#endif

#include "wx/dnd.h"        // for wxFileDropTarget
#include "wx/filename.h"   // for wxFileName
#include "wx/dir.h"        // for wxDir
#include "wx/clipbrd.h"    // for wxTheClipboard
#if wxUSE_TOOLTIPS
   #include "wx/tooltip.h" // for wxToolTip
#endif

#include "bigint.h"
#include "lifealgo.h"
#include "qlifealgo.h"
#include "hlifealgo.h"

#include "wxgolly.h"       // for wxGetApp, statusptr, viewptr, bigview
#include "wxutils.h"       // for Warning, Fatal, etc
#include "wxprefs.h"       // for gollydir, datadir, SavePrefs, etc
#include "wxinfo.h"        // for ShowInfo, GetInfoFrame
#include "wxhelp.h"        // for ShowHelp, GetHelpFrame
#include "wxstatus.h"      // for statusptr->...
#include "wxview.h"        // for viewptr->...
#include "wxrender.h"      // for InitDrawingData, DestroyDrawingData, etc
#include "wxedit.h"        // for CreateEditBar, EditBarHeight, etc
#include "wxscript.h"      // for inscript
#include "wxalgos.h"       // for algo_type, algomenu, algomenupop
#include "wxlayer.h"       // for AddLayer, MAX_LAYERS, currlayer
#include "wxundo.h"        // for currlayer->undoredo->...
#include "wxtimeline.h"    // for CreateTimelineBar, TimelineExists, etc
#include "wxmain.h"

#ifdef __WXMAC__
   #include <Carbon/Carbon.h>    // for GetCurrentProcess, etc
#endif

// -----------------------------------------------------------------------------

// one-shot timer to fix problems in wxMac and wxGTK -- see OnOneTimer;
// must be static global because it's used in DnDFile::OnDropFiles
static wxTimer* onetimer;

#ifdef __WXMSW__
static bool call_unselect = false;        // OnIdle needs to call Unselect?
static wxString editpath = wxEmptyString; // OnIdle calls EditFile if this isn't empty
static bool ignore_selection = false;     // ignore spurious selection?
#endif

static bool call_close = false;           // OnIdle needs to call Close?
static bool edit_file = false;            // edit the clicked file?

// -----------------------------------------------------------------------------

// ids for bitmap buttons in tool bar
enum {
   START_TOOL = 0,
   STOP_TOOL,
   RESET_TOOL,
   ALGO_TOOL,
   AUTOFIT_TOOL,
   HYPER_TOOL,
   NEW_TOOL,
   OPEN_TOOL,
   SAVE_TOOL,
   PATTERNS_TOOL,
   SCRIPTS_TOOL,
   INFO_TOOL,
   HELP_TOOL,
   NUM_BUTTONS    // must be last
};

#ifdef __WXMSW__
   // bitmaps are loaded via .rc file
#else
   // bitmaps for tool bar buttons
   #include "bitmaps/play.xpm"
   #include "bitmaps/stop.xpm"
   #include "bitmaps/reset.xpm"
   #include "bitmaps/algo.xpm"
   #include "bitmaps/autofit.xpm"
   #include "bitmaps/hyper.xpm"
   #include "bitmaps/new.xpm"
   #include "bitmaps/open.xpm"
   #include "bitmaps/save.xpm"
   #include "bitmaps/patterns.xpm"
   #include "bitmaps/scripts.xpm"
   #include "bitmaps/info.xpm"
   #include "bitmaps/help.xpm"
   // bitmaps for down state of toggle buttons
   #include "bitmaps/autofit_down.xpm"
   #include "bitmaps/hyper_down.xpm"
   #include "bitmaps/patterns_down.xpm"
   #include "bitmaps/scripts_down.xpm"
#endif

// -----------------------------------------------------------------------------

// Define our own vertical tool bar to avoid bugs and limitations in wxToolBar:

// derive from wxPanel so we get current theme's background color on Windows
class ToolBar : public wxPanel
{
public:
   ToolBar(wxWindow* parent, wxCoord xorg, wxCoord yorg, int wd, int ht);
   ~ToolBar() {}

   // add a bitmap button to tool bar
   void AddButton(int id, const wxString& tip);

   // add a vertical gap between buttons
   void AddSeparator();
   
   // enable/disable button
   void EnableButton(int id, bool enable);

   // set state of start/stop button
   void SetStartStopButton();
   
   // set state of a toggle button
   void SelectButton(int id, bool select);

   // detect press and release of a bitmap button
   void OnButtonDown(wxMouseEvent& event);
   void OnButtonUp(wxMouseEvent& event);
   void OnKillFocus(wxFocusEvent& event);
   
private:
   // any class wishing to process wxWidgets events must use this macro
   DECLARE_EVENT_TABLE()

   // event handlers
   void OnPaint(wxPaintEvent& event);
   void OnMouseDown(wxMouseEvent& event);
   void OnButton(wxCommandEvent& event);
   
   // bitmaps for normal or down state
   wxBitmap normtool[NUM_BUTTONS];
   wxBitmap downtool[NUM_BUTTONS];

   #ifdef __WXMSW__
      // on Windows we need bitmaps for disabled buttons
      wxBitmap disnormtool[NUM_BUTTONS];
      wxBitmap disdowntool[NUM_BUTTONS];
   #endif
   
   // remember state of toggle buttons to avoid unnecessary drawing;
   // 0 = not yet initialized, 1 = selected, -1 = not selected
   int buttstate[NUM_BUTTONS];
   
   // positioning data used by AddButton and AddSeparator
   int ypos, xpos, smallgap, biggap;
};

BEGIN_EVENT_TABLE(ToolBar, wxPanel)
   EVT_PAINT            (           ToolBar::OnPaint)
   EVT_LEFT_DOWN        (           ToolBar::OnMouseDown)
   EVT_BUTTON           (wxID_ANY,  ToolBar::OnButton)
END_EVENT_TABLE()

ToolBar* toolbarptr = NULL;      // global pointer to tool bar
const int toolbarwd = 32;        // width of (vertical) tool bar

// tool bar buttons (must be global to use Connect/Disconnect on Windows)
wxBitmapButton* tbbutt[NUM_BUTTONS];

// width and height of bitmap buttons
#if defined(__WXOSX_COCOA__) || defined(__WXGTK__)
   const int BUTTON_WD = 28;
   const int BUTTON_HT = 28;
#else
   const int BUTTON_WD = 24;
   const int BUTTON_HT = 24;
#endif

// -----------------------------------------------------------------------------

ToolBar::ToolBar(wxWindow* parent, wxCoord xorg, wxCoord yorg, int wd, int ht)
   : wxPanel(parent, wxID_ANY, wxPoint(xorg,yorg), wxSize(wd,ht),
             wxNO_FULL_REPAINT_ON_RESIZE)
{
   #ifdef __WXGTK__
      // avoid erasing background on GTK+
      SetBackgroundStyle(wxBG_STYLE_CUSTOM);
   #endif

   // init bitmaps for normal state
   normtool[START_TOOL] =     wxBITMAP(play);
   normtool[STOP_TOOL] =      wxBITMAP(stop);
   normtool[RESET_TOOL] =     wxBITMAP(reset);
   normtool[ALGO_TOOL] =      wxBITMAP(algo);
   normtool[AUTOFIT_TOOL] =   wxBITMAP(autofit);
   normtool[HYPER_TOOL] =     wxBITMAP(hyper);
   normtool[NEW_TOOL] =       wxBITMAP(new);
   normtool[OPEN_TOOL] =      wxBITMAP(open);
   normtool[SAVE_TOOL] =      wxBITMAP(save);
   normtool[PATTERNS_TOOL] =  wxBITMAP(patterns);
   normtool[SCRIPTS_TOOL] =   wxBITMAP(scripts);
   normtool[INFO_TOOL] =      wxBITMAP(info);
   normtool[HELP_TOOL] =      wxBITMAP(help);
   
   // toggle buttons also have a down state
   downtool[AUTOFIT_TOOL] =   wxBITMAP(autofit_down);
   downtool[HYPER_TOOL] =     wxBITMAP(hyper_down);
   downtool[PATTERNS_TOOL] =  wxBITMAP(patterns_down);
   downtool[SCRIPTS_TOOL] =   wxBITMAP(scripts_down);

   #ifdef __WXMSW__
      for (int i = 0; i < NUM_BUTTONS; i++) {
         CreatePaleBitmap(normtool[i], disnormtool[i]);
      }
      CreatePaleBitmap(downtool[AUTOFIT_TOOL],  disdowntool[AUTOFIT_TOOL]);
      CreatePaleBitmap(downtool[HYPER_TOOL],    disdowntool[HYPER_TOOL]);
      CreatePaleBitmap(downtool[PATTERNS_TOOL], disdowntool[PATTERNS_TOOL]);
      CreatePaleBitmap(downtool[SCRIPTS_TOOL],  disdowntool[SCRIPTS_TOOL]);
   #endif

   for (int i = 0; i < NUM_BUTTONS; i++) {
      buttstate[i] = 0;
   }

   // init position variables used by AddButton and AddSeparator
   #ifdef __WXGTK__
      // buttons are a different size in wxGTK
      xpos = 2;
      ypos = 2;
      smallgap = 6;
   #else
      xpos = (32 - BUTTON_WD) / 2;
      ypos = (32 - BUTTON_HT) / 2;
      smallgap = 4;
   #endif
   biggap = 16;
}

// -----------------------------------------------------------------------------

void ToolBar::OnPaint(wxPaintEvent& WXUNUSED(event))
{
   wxPaintDC dc(this);

   int wd, ht;
   GetClientSize(&wd, &ht);
   if (wd < 1 || ht < 1 || !showtool) return;
      
   wxRect r = wxRect(0, 0, wd, ht);   
   #ifdef __WXMSW__
      dc.Clear();
      // draw gray line at top edge
      dc.SetPen(*wxGREY_PEN);
      dc.DrawLine(0, 0, r.width, 0);
      dc.SetPen(wxNullPen);
   #else
      // draw gray line at right edge
      #ifdef __WXMAC__
         wxBrush brush(wxColor(202,202,202));
         FillRect(dc, r, brush);
         wxPen linepen(wxColor(140,140,140));
         dc.SetPen(linepen);
      #else
         dc.SetPen(*wxLIGHT_GREY_PEN);
      #endif
      dc.DrawLine(r.GetRight(), 0, r.GetRight(), r.height);
      dc.SetPen(wxNullPen);
   #endif
}

// -----------------------------------------------------------------------------

void ToolBar::OnMouseDown(wxMouseEvent& WXUNUSED(event))
{
   // this is NOT called if user clicks a tool bar button;
   // on Windows we need to reset keyboard focus to viewport window
   viewptr->SetFocus();
}

// -----------------------------------------------------------------------------

void ToolBar::OnButton(wxCommandEvent& event)
{
   int id = event.GetId();

   int cmdid;
   switch (id) {
      case START_TOOL:     cmdid = ID_START; break;
      case RESET_TOOL:     cmdid = ID_RESET; break;
      case ALGO_TOOL:      return;                    // handled in OnButtonDown
      case AUTOFIT_TOOL:   cmdid = ID_AUTO; break;
      case HYPER_TOOL:     cmdid = ID_HYPER; break;
      case NEW_TOOL:       cmdid = wxID_NEW; break;
      case OPEN_TOOL:      cmdid = wxID_OPEN; break;
      case SAVE_TOOL:      cmdid = wxID_SAVE; break;
      case PATTERNS_TOOL:  cmdid = ID_SHOW_PATTERNS; break;
      case SCRIPTS_TOOL:   cmdid = ID_SHOW_SCRIPTS; break;
      case INFO_TOOL:      cmdid = ID_INFO; break;
      case HELP_TOOL:      cmdid = ID_HELP_BUTT; break;
      default:             Warning(_("Unexpected button id!")); return;
   }
   
   // call MainFrame::OnMenu after OnButton finishes;
   // this avoids start/stop button problem in GTK app
   wxCommandEvent cmdevt(wxEVT_COMMAND_MENU_SELECTED, cmdid);
   wxPostEvent(mainptr->GetEventHandler(), cmdevt);

   // avoid weird bug on Mac where all buttons can be disabled after hitting
   // RESET_TOOL button *and* the "All controls" option is ticked in
   // System Prefs > Keyboard & Mouse > Keyboard Shortcuts
   // (might also fix similar problem on Windows)
   viewptr->SetFocus();
}

// -----------------------------------------------------------------------------

void ToolBar::OnKillFocus(wxFocusEvent& event)
{
   int id = event.GetId();
   tbbutt[id]->SetFocus();   // don't let button lose focus
}

// -----------------------------------------------------------------------------

void ToolBar::OnButtonDown(wxMouseEvent& event)
{
   // a tool bar button has been pressed
   int id = event.GetId();
   
   #ifdef __WXMSW__
      // connect a handler that keeps focus with the pressed button
      if (id != ALGO_TOOL)
         tbbutt[id]->Connect(id, wxEVT_KILL_FOCUS,
                             wxFocusEventHandler(ToolBar::OnKillFocus));
   #endif

   #ifdef __WXMAC__
      // close any open tool tip window (probably wxMac bug)
      wxToolTip::RemoveToolTips();
   #endif

   // we want pop-up menu to appear as soon as ALGO_TOOL button is pressed
   //!!! doesn't work 100% correctly on Windows
   if (id == ALGO_TOOL) {
      // we use algomenupop here rather than algomenu to avoid assert messages in wx 2.9+
      #ifdef __WXMSW__
         tbbutt[id]->PopupMenu(algomenupop, 0, 25);
         // fix prob on Win (almost -- button-up doesn't always close menu)
         viewptr->SetFocus();
         return;
      #else
         tbbutt[id]->PopupMenu(algomenupop, 0, 30);
      #endif
      
      #ifdef __WXOSX_COCOA__
         viewptr->SetFocus();
         // don't call event.Skip() otherwise algo button will remain selected
         return;
      #endif
   }

   event.Skip();
}

// -----------------------------------------------------------------------------

void ToolBar::OnButtonUp(wxMouseEvent& event)
{
   // a tool bar button has been released (only called in wxMSW)
   int id = event.GetId();

   wxPoint pt = tbbutt[id]->ScreenToClient( wxGetMousePosition() );

   int wd, ht;
   tbbutt[id]->GetClientSize(&wd, &ht);
   wxRect r(0, 0, wd, ht);

   // diconnect kill-focus handler
   if (id != ALGO_TOOL)
      tbbutt[id]->Disconnect(id, wxEVT_KILL_FOCUS,
                             wxFocusEventHandler(ToolBar::OnKillFocus));
   viewptr->SetFocus();

   if ( r.Contains(pt) ) {
      // call OnButton
      wxCommandEvent buttevt(wxEVT_COMMAND_BUTTON_CLICKED, id);
      buttevt.SetEventObject(tbbutt[id]);
      tbbutt[id]->GetEventHandler()->ProcessEvent(buttevt);
   }
}

// -----------------------------------------------------------------------------

void ToolBar::AddButton(int id, const wxString& tip)
{
   tbbutt[id] = new wxBitmapButton(this, id, normtool[id], wxPoint(xpos,ypos), wxSize(BUTTON_WD, BUTTON_HT));
   if (tbbutt[id] == NULL) {
      Fatal(_("Failed to create tool bar button!"));
   } else {
      ypos += BUTTON_HT + smallgap;
      tbbutt[id]->SetToolTip(tip);
      #ifdef __WXMSW__
         // fix problem with tool bar buttons when generating/inscript
         // due to focus being changed to viewptr
         tbbutt[id]->Connect(id, wxEVT_LEFT_DOWN, wxMouseEventHandler(ToolBar::OnButtonDown));
         tbbutt[id]->Connect(id, wxEVT_LEFT_UP, wxMouseEventHandler(ToolBar::OnButtonUp));
      #else
         // allow pop-up menu to appear as soon as ALGO_TOOL button is pressed
         tbbutt[id]->Connect(id, wxEVT_LEFT_DOWN, wxMouseEventHandler(ToolBar::OnButtonDown));
      #endif
   }
}

// -----------------------------------------------------------------------------

void ToolBar::AddSeparator()
{
   ypos += biggap - smallgap;
}

// -----------------------------------------------------------------------------

void ToolBar::EnableButton(int id, bool enable)
{
   if (enable == tbbutt[id]->IsEnabled()) return;

   #ifdef __WXMSW__
      if (id == START_TOOL && (inscript || mainptr->generating)) {
         tbbutt[id]->SetBitmapDisabled(disnormtool[STOP_TOOL]);
         
      } else if (id == AUTOFIT_TOOL && currlayer->autofit) {
         tbbutt[id]->SetBitmapDisabled(disdowntool[id]);
         
      } else if (id == HYPER_TOOL && currlayer->hyperspeed) {
         tbbutt[id]->SetBitmapDisabled(disdowntool[id]);
         
      } else if (id == PATTERNS_TOOL && showpatterns) {
         tbbutt[id]->SetBitmapDisabled(disdowntool[id]);
         
      } else if (id == SCRIPTS_TOOL && showscripts) {
         tbbutt[id]->SetBitmapDisabled(disdowntool[id]);
         
      } else {
         tbbutt[id]->SetBitmapDisabled(disnormtool[id]);
      }
   #endif

   tbbutt[id]->Enable(enable);
}

// -----------------------------------------------------------------------------

void ToolBar::SetStartStopButton()
{
   if (inscript || mainptr->generating) {
      // show stop bitmap
      if (buttstate[START_TOOL] == 1) return;
      buttstate[START_TOOL] = 1;
      tbbutt[START_TOOL]->SetBitmapLabel(normtool[STOP_TOOL]);
      if (inscript)
         tbbutt[START_TOOL]->SetToolTip(_("Stop script"));
      else
         tbbutt[START_TOOL]->SetToolTip(_("Stop generating"));
   } else {
      // show start bitmap
      if (buttstate[START_TOOL] == -1) return;
      buttstate[START_TOOL] = -1;
      tbbutt[START_TOOL]->SetBitmapLabel(normtool[START_TOOL]);
      tbbutt[START_TOOL]->SetToolTip(_("Start generating"));
   }

   tbbutt[START_TOOL]->Refresh(false);
}

// -----------------------------------------------------------------------------

void ToolBar::SelectButton(int id, bool select)
{
   if (select) {
      if (buttstate[id] == 1) return;
      buttstate[id] = 1;
      tbbutt[id]->SetBitmapLabel(downtool[id]);
   } else {
      if (buttstate[id] == -1) return;
      buttstate[id] = -1;
      tbbutt[id]->SetBitmapLabel(normtool[id]);
   }

   tbbutt[id]->Refresh(false);
}

// -----------------------------------------------------------------------------

void MainFrame::CreateToolbar()
{
   int wd, ht;
   GetClientSize(&wd, &ht);

   toolbarptr = new ToolBar(this, 0, 0, toolbarwd, ht);
   if (toolbarptr == NULL) Fatal(_("Failed to create tool bar!"));

   // add buttons to tool bar
   toolbarptr->AddButton(START_TOOL,      _("Start generating"));
   toolbarptr->AddButton(RESET_TOOL,      _("Reset"));
   toolbarptr->AddSeparator();
   toolbarptr->AddButton(ALGO_TOOL,       _("Set algorithm"));
   toolbarptr->AddButton(AUTOFIT_TOOL,    _("Auto fit"));
   toolbarptr->AddButton(HYPER_TOOL,      _("Hyperspeed"));
   toolbarptr->AddSeparator();
   toolbarptr->AddButton(NEW_TOOL,        _("New pattern"));
   toolbarptr->AddButton(OPEN_TOOL,       _("Open pattern"));
   toolbarptr->AddButton(SAVE_TOOL,       _("Save pattern"));
   toolbarptr->AddSeparator();
   toolbarptr->AddButton(PATTERNS_TOOL,   _("Show/hide patterns"));
   toolbarptr->AddButton(SCRIPTS_TOOL,    _("Show/hide scripts"));
   toolbarptr->AddSeparator();
   toolbarptr->AddButton(INFO_TOOL,       _("Show pattern information"));
   toolbarptr->AddButton(HELP_TOOL,       _("Show help window"));
      
   toolbarptr->Show(showtool);
}

// -----------------------------------------------------------------------------

void MainFrame::UpdateToolBar(bool active)
{
   // update tool bar buttons according to the current state
   if (toolbarptr && showtool) {
      if (viewptr->waitingforclick) active = false;
      bool timeline = TimelineExists();

      // set state of start/stop button
      toolbarptr->SetStartStopButton();

      // set state of toggle buttons
      toolbarptr->SelectButton(AUTOFIT_TOOL,    currlayer->autofit);
      toolbarptr->SelectButton(HYPER_TOOL,      currlayer->hyperspeed);
      toolbarptr->SelectButton(PATTERNS_TOOL,   showpatterns);
      toolbarptr->SelectButton(SCRIPTS_TOOL,    showscripts);
      
      toolbarptr->EnableButton(START_TOOL,      active && !timeline);
      toolbarptr->EnableButton(RESET_TOOL,      active && !timeline && !inscript && (generating ||
                                                currlayer->algo->getGeneration() > currlayer->startgen));
      toolbarptr->EnableButton(ALGO_TOOL,       active && !timeline && !inscript);
      toolbarptr->EnableButton(AUTOFIT_TOOL,    active);
      toolbarptr->EnableButton(HYPER_TOOL,      active && !timeline);
      toolbarptr->EnableButton(NEW_TOOL,        active && !inscript);
      toolbarptr->EnableButton(OPEN_TOOL,       active && !inscript);
      toolbarptr->EnableButton(SAVE_TOOL,       active && !inscript);
      toolbarptr->EnableButton(PATTERNS_TOOL,   active);
      toolbarptr->EnableButton(SCRIPTS_TOOL,    active);
      toolbarptr->EnableButton(INFO_TOOL,       active && !currlayer->currfile.IsEmpty());
      toolbarptr->EnableButton(HELP_TOOL,       active);
   }
}

// -----------------------------------------------------------------------------

bool MainFrame::ClipboardHasText()
{
   bool hastext = false;
   #ifdef __WXGTK__
      // avoid re-entrancy bug in wxGTK 2.9.x
      if (wxTheClipboard->IsOpened()) return false;
   #endif
   if (wxTheClipboard->Open()) {
      hastext = wxTheClipboard->IsSupported(wxDF_TEXT);
      if (!hastext) {
         // we'll try to convert bitmap data to text pattern
         hastext = wxTheClipboard->IsSupported(wxDF_BITMAP);
      }
      wxTheClipboard->Close();
   }
   return hastext;
}

// -----------------------------------------------------------------------------

void MainFrame::EnableAllMenus(bool enable)
{
   #if defined(__WXMAC__) && !defined(__WXOSX_COCOA__)
      // enable/disable all menus, including Help menu and items in app menu
      if (enable)
         EndAppModalStateForWindow( (OpaqueWindowPtr*)this->MacGetWindowRef() );
      else
         BeginAppModalStateForWindow( (OpaqueWindowPtr*)this->MacGetWindowRef() );
   #else
      wxMenuBar* mbar = GetMenuBar();
      if (mbar) {
         int count = mbar->GetMenuCount();
         int i;
         for (i = 0; i < count; i++) {
            mbar->EnableTop(i, enable);
         }
         #ifdef __WXOSX_COCOA__
            // enable/disable items in app menu
            //!!! these fail to disable -- wxOSX bug???
            mbar->Enable(wxID_ABOUT, enable);
            mbar->Enable(wxID_PREFERENCES, enable);
            mbar->Enable(wxID_EXIT, enable);
         #endif
      }
   #endif
}

// -----------------------------------------------------------------------------

void MainFrame::UpdateMenuItems(bool active)
{
   // update menu bar items according to the given state
   wxMenuBar* mbar = GetMenuBar();
   if (mbar) {
      bool textinclip = ClipboardHasText();
      bool selexists = viewptr->SelectionExists();
      bool timeline = TimelineExists();

      if (viewptr->waitingforclick) active = false;
      
      mbar->Enable(wxID_NEW,           active && !inscript);
      mbar->Enable(wxID_OPEN,          active && !inscript);
      mbar->Enable(ID_OPEN_CLIP,       active && !inscript && textinclip);
      mbar->Enable(ID_OPEN_RECENT,     active && !inscript && numpatterns > 0);
      mbar->Enable(ID_SHOW_PATTERNS,   active);
      mbar->Enable(ID_PATTERN_DIR,     active);
      mbar->Enable(wxID_SAVE,          active && !inscript);
      mbar->Enable(ID_SAVE_XRLE,       active);
      mbar->Enable(ID_RUN_SCRIPT,      active && !timeline && !inscript);
      mbar->Enable(ID_RUN_CLIP,        active && !timeline && !inscript && textinclip);
      mbar->Enable(ID_RUN_RECENT,      active && !timeline && !inscript && numscripts > 0);
      mbar->Enable(ID_SHOW_SCRIPTS,    active);
      mbar->Enable(ID_SCRIPT_DIR,      active);
      // safer not to allow prefs dialog while script is running???
      // mbar->Enable(wxID_PREFERENCES,   !inscript);

      bool can_undo = active && !timeline && currlayer->undoredo->CanUndo();
      bool can_redo = active && !timeline && currlayer->undoredo->CanRedo();
      #if defined(__WXMAC__) && !defined(__WXOSX_COCOA__)
         // need this stupidity to avoid wxMac bug after modal dialog closes (eg. Set Rule)
         // and force items to appear correctly enabled/disabled
         mbar->Enable(ID_UNDO, !can_undo);
         mbar->Enable(ID_REDO, !can_redo);
      #endif

      mbar->Enable(ID_UNDO,            can_undo);
      mbar->Enable(ID_REDO,            can_redo);
      mbar->Enable(ID_NO_UNDO,         active && !inscript);
      mbar->Enable(ID_CUT,             active && !timeline && !inscript && selexists);
      mbar->Enable(ID_COPY,            active && !inscript && selexists);
      mbar->Enable(ID_CLEAR,           active && !timeline && !inscript && selexists);
      mbar->Enable(ID_OUTSIDE,         active && !timeline && !inscript && selexists);
      mbar->Enable(ID_PASTE,           active && !timeline && !inscript && textinclip);
      mbar->Enable(ID_PASTE_SEL,       active && !timeline && !inscript && textinclip && selexists);
      mbar->Enable(ID_PLOCATION,       active);
      mbar->Enable(ID_PMODE,           active);
      mbar->Enable(ID_SELECTALL,       active && !inscript);
      mbar->Enable(ID_REMOVE,          active && !inscript && selexists);
      mbar->Enable(ID_SHRINK,          active && !inscript && selexists);
      mbar->Enable(ID_RANDOM,          active && !timeline && !inscript && selexists);
      mbar->Enable(ID_FLIPTB,          active && !timeline && !inscript && selexists);
      mbar->Enable(ID_FLIPLR,          active && !timeline && !inscript && selexists);
      mbar->Enable(ID_ROTATEC,         active && !timeline && !inscript && selexists);
      mbar->Enable(ID_ROTATEA,         active && !timeline && !inscript && selexists);
      mbar->Enable(ID_CMODE,           active);

      if (inscript) {
         // don't use DO_STARTSTOP key to abort a running script
         #ifdef __WXMAC__
            // on Mac we need to clear the accelerator first because "\tEscape" doesn't really
            // change the accelerator (it just looks like it does!) -- this is because escape
            // (key code 27) is used by SetItemCmd to indicate the item has a submenu;
            // see UMASetMenuItemShortcut in wx/src/mac/carbon/uma.cpp
            mbar->SetLabel(ID_START, _("x"));
         #endif
         mbar->SetLabel(ID_START, _("Stop Script\tEscape"));
      } else if (generating) {
         mbar->SetLabel(ID_START, _("Stop Generating") + GetAccelerator(DO_STARTSTOP));
      } else if (timeline && !currlayer->algo->isrecording()) {
         if (TimelineIsPlaying())
            mbar->SetLabel(ID_START, _("Stop Playing Timeline") + GetAccelerator(DO_STARTSTOP));
         else
            mbar->SetLabel(ID_START, _("Start Playing Timeline") + GetAccelerator(DO_STARTSTOP));
      } else {
         mbar->SetLabel(ID_START, _("Start Generating") + GetAccelerator(DO_STARTSTOP));
      }
      
      if (currlayer->algo->isrecording()) {
         mbar->SetLabel(ID_RECORD, _("Stop Recording") + GetAccelerator(DO_RECORD));
      } else {
         mbar->SetLabel(ID_RECORD, _("Start Recording") + GetAccelerator(DO_RECORD));
      }

      mbar->Enable(ID_START,        active && !currlayer->algo->isrecording());
      mbar->Enable(ID_NEXT,         active && !timeline && !generating && !inscript);
      mbar->Enable(ID_STEP,         active && !timeline && !generating && !inscript);
      mbar->Enable(ID_RESET,        active && !timeline && !inscript && (generating ||
                                    currlayer->algo->getGeneration() > currlayer->startgen));
      mbar->Enable(ID_SETGEN,       active && !timeline && !inscript);
      mbar->Enable(ID_FASTER,       active);
      mbar->Enable(ID_SLOWER,       active /* && currlayer->currexpo > minexpo */);
                                           // don't do this test because Win users won't hear beep
      mbar->Enable(ID_SETBASE,      active && !timeline && !inscript);
      mbar->Enable(ID_AUTO,         active);
      mbar->Enable(ID_HYPER,        active && !timeline);
      mbar->Enable(ID_HINFO,        active);
      mbar->Enable(ID_RECORD,       active && !inscript && currlayer->algo->hyperCapable());
      mbar->Enable(ID_DELTIME,      active && !inscript && timeline && !currlayer->algo->isrecording());
      mbar->Enable(ID_SETRULE,      active && !timeline && !inscript);
      mbar->Enable(ID_SETALGO,      active && !timeline && !inscript);

      mbar->Enable(ID_FULL,         active);
      mbar->Enable(ID_FIT,          active);
      mbar->Enable(ID_FIT_SEL,      active && selexists);
      mbar->Enable(ID_MIDDLE,       active);
      mbar->Enable(ID_RESTORE00,    active && (currlayer->originx != bigint::zero ||
                                               currlayer->originy != bigint::zero));
      mbar->Enable(wxID_ZOOM_IN,    active /* && viewptr->GetMag() < MAX_MAG */);
                                           // don't do this test because Win users won't hear beep
      mbar->Enable(wxID_ZOOM_OUT,   active);
      mbar->Enable(ID_SET_SCALE,    active);
      mbar->Enable(ID_TOOL_BAR,     active);
      mbar->Enable(ID_LAYER_BAR,    active);
      mbar->Enable(ID_EDIT_BAR,     active);
      mbar->Enable(ID_ALL_STATES,   active);
      mbar->Enable(ID_STATUS_BAR,   active);
      mbar->Enable(ID_EXACT,        active);
      mbar->Enable(ID_GRID,         active);
      mbar->Enable(ID_ICONS,        active);
      mbar->Enable(ID_INVERT,       active);
      #if defined(__WXMAC__) || defined(__WXGTK__)
         // windows on Mac OS X and GTK+ 2.0 are automatically buffered
         mbar->Enable(ID_BUFF,      false);
         mbar->Check(ID_BUFF,       true);
      #else
         mbar->Enable(ID_BUFF,      active);
         mbar->Check(ID_BUFF,       buffered);
      #endif
      mbar->Enable(ID_TIMELINE,     active);
      mbar->Enable(ID_INFO,         !currlayer->currfile.IsEmpty());

      mbar->Enable(ID_ADD_LAYER,    active && !inscript && numlayers < MAX_LAYERS);
      mbar->Enable(ID_CLONE,        active && !inscript && numlayers < MAX_LAYERS);
      mbar->Enable(ID_DUPLICATE,    active && !inscript && numlayers < MAX_LAYERS);
      mbar->Enable(ID_DEL_LAYER,    active && !inscript && numlayers > 1);
      mbar->Enable(ID_DEL_OTHERS,   active && !inscript && numlayers > 1);
      mbar->Enable(ID_MOVE_LAYER,   active && !inscript && numlayers > 1);
      mbar->Enable(ID_NAME_LAYER,   active && !inscript);
      mbar->Enable(ID_SET_COLORS,   active && !inscript);
      mbar->Enable(ID_SYNC_VIEW,    active);
      mbar->Enable(ID_SYNC_CURS,    active);
      mbar->Enable(ID_STACK,        active);
      mbar->Enable(ID_TILE,         active);
      for (int i = 0; i < numlayers; i++)
         mbar->Enable(ID_LAYER0 + i, active && CanSwitchLayer(i));

      // tick/untick menu items created using AppendCheckItem
      mbar->Check(ID_SAVE_XRLE,     savexrle);
      mbar->Check(ID_SHOW_PATTERNS, showpatterns);
      mbar->Check(ID_SHOW_SCRIPTS,  showscripts);
      mbar->Check(ID_NO_UNDO,       !allowundo);
      mbar->Check(ID_AUTO,       currlayer->autofit);
      mbar->Check(ID_HYPER,      currlayer->hyperspeed);
      mbar->Check(ID_HINFO,      currlayer->showhashinfo);
      mbar->Check(ID_TOOL_BAR,   showtool);
      mbar->Check(ID_LAYER_BAR,  showlayer);
      mbar->Check(ID_EDIT_BAR,   showedit);
      mbar->Check(ID_ALL_STATES, showallstates);
      mbar->Check(ID_STATUS_BAR, showstatus);
      mbar->Check(ID_EXACT,      showexact);
      mbar->Check(ID_GRID,       showgridlines);
      mbar->Check(ID_ICONS,      showicons);
      mbar->Check(ID_INVERT,     swapcolors);
      mbar->Check(ID_TIMELINE,   showtimeline);
      mbar->Check(ID_PL_TL,      plocation == TopLeft);
      mbar->Check(ID_PL_TR,      plocation == TopRight);
      mbar->Check(ID_PL_BR,      plocation == BottomRight);
      mbar->Check(ID_PL_BL,      plocation == BottomLeft);
      mbar->Check(ID_PL_MID,     plocation == Middle);
      mbar->Check(ID_PM_AND,     pmode == And);
      mbar->Check(ID_PM_COPY,    pmode == Copy);
      mbar->Check(ID_PM_OR,      pmode == Or);
      mbar->Check(ID_PM_XOR,     pmode == Xor);
      mbar->Check(ID_DRAW,       currlayer->curs == curs_pencil);
      mbar->Check(ID_PICK,       currlayer->curs == curs_pick);
      mbar->Check(ID_SELECT,     currlayer->curs == curs_cross);
      mbar->Check(ID_MOVE,       currlayer->curs == curs_hand);
      mbar->Check(ID_ZOOMIN,     currlayer->curs == curs_zoomin);
      mbar->Check(ID_ZOOMOUT,    currlayer->curs == curs_zoomout);
      mbar->Check(ID_SCALE_1,    viewptr->GetMag() == 0);
      mbar->Check(ID_SCALE_2,    viewptr->GetMag() == 1);
      mbar->Check(ID_SCALE_4,    viewptr->GetMag() == 2);
      mbar->Check(ID_SCALE_8,    viewptr->GetMag() == 3);
      mbar->Check(ID_SCALE_16,   viewptr->GetMag() == 4);
      mbar->Check(ID_SYNC_VIEW,  syncviews);
      mbar->Check(ID_SYNC_CURS,  synccursors);
      mbar->Check(ID_STACK,      stacklayers);
      mbar->Check(ID_TILE,       tilelayers);
      for (int i = 0; i < NumAlgos(); i++) {
         mbar->Check(ID_ALGO0 + i, currlayer->algtype == i);
         // keep algomenupop in sync with algomenu
         algomenupop->Check(ID_ALGO0 + i, currlayer->algtype == i);
      }
      for (int i = 0; i < numlayers; i++) {
         mbar->Check(ID_LAYER0 + i, currindex == i);
      }
   }
}

// -----------------------------------------------------------------------------

void MainFrame::UpdateUserInterface(bool active)
{
   UpdateToolBar(active);
   UpdateLayerBar(active);
   UpdateEditBar(active);
   UpdateTimelineBar(active);
   UpdateMenuItems(active);
   viewptr->CheckCursor(active);
   statusptr->CheckMouseLocation(active);

   #ifdef __WXMSW__
      // ensure viewport window has keyboard focus if main window is active
      if (active) viewptr->SetFocus();
   #endif
}

// -----------------------------------------------------------------------------

// update everything in main window, and menu bar and cursor
void MainFrame::UpdateEverything()
{
   if (IsIconized()) {
      // main window has been minimized, so only update menu bar items
      UpdateMenuItems(false);
      return;
   }

   // update all bars, menus and cursor
   UpdateUserInterface(IsActive());

   if (inscript) {
      // make sure scroll bars are accurate while running script
      bigview->UpdateScrollBars();
      return;
   }

   int wd, ht;
   GetClientSize(&wd, &ht);      // includes status bar and viewport

   if (wd > 0 && ht > statusptr->statusht) {
      UpdateView();
      bigview->UpdateScrollBars();
   }
   
   if (wd > 0 && ht > 0 && showstatus) {
      statusptr->Refresh(false);
      statusptr->Update();
   }
}

// -----------------------------------------------------------------------------

// only update viewport and status bar
void MainFrame::UpdatePatternAndStatus()
{
   if (inscript || currlayer->undoredo->doingscriptchanges) return;

   if (!IsIconized()) {
      UpdateView();
      if (showstatus) {
         statusptr->CheckMouseLocation(IsActive());
         statusptr->Refresh(false);
         statusptr->Update();
      }
   }
}

// -----------------------------------------------------------------------------

// only update status bar
void MainFrame::UpdateStatus()
{
   if (inscript || currlayer->undoredo->doingscriptchanges) return;

   if (!IsIconized()) {
      if (showstatus) {
         statusptr->CheckMouseLocation(IsActive());
         statusptr->Refresh(false);
         statusptr->Update();
      }
   }
}

// -----------------------------------------------------------------------------

void MainFrame::SimplifyTree(wxString& indir, wxTreeCtrl* treectrl, wxTreeItemId root)
{
   // delete old tree (except root)
   treectrl->DeleteChildren(root);
   
   // remove any terminating separator
   wxString dir = indir;
   if (dir.Last() == wxFILE_SEP_PATH) dir.Truncate(dir.Length()-1);

   // append dir as only child
   wxDirItemData* diritem = new wxDirItemData(dir, dir, true);
   wxTreeItemId id;
   id = treectrl->AppendItem(root, dir.AfterLast(wxFILE_SEP_PATH), 0, 0, diritem);
   if ( diritem->HasFiles() || diritem->HasSubDirs() ) {
      treectrl->SetItemHasChildren(id);
      treectrl->Expand(id);
      
      // nicer to expand Perl & Python subdirs inside Scripts
      if ( dir == gollydir + _("Scripts") ) {
         wxTreeItemId child;
         wxTreeItemIdValue cookie;
         child = treectrl->GetFirstChild(id, cookie);
         while ( child.IsOk() ) {
            wxString name = treectrl->GetItemText(child);
            if ( name == _("Perl") || name == _("Python") ) {
               treectrl->Expand(child);
            }
            child = treectrl->GetNextChild(id, cookie);
         }
      }
      
      #ifndef __WXMSW__
         // causes crash on Windows
         treectrl->ScrollTo(root);
      #endif
   }
}

// -----------------------------------------------------------------------------

void MainFrame::DeselectTree(wxTreeCtrl* treectrl, wxTreeItemId root)
{
   // recursively traverse tree and reset each file item background to white
   wxTreeItemIdValue cookie;
   wxTreeItemId id = treectrl->GetFirstChild(root, cookie);
   while ( id.IsOk() ) {
      if ( treectrl->ItemHasChildren(id) ) {
         DeselectTree(treectrl, id);
      } else {
         wxColor currcolor = treectrl->GetItemBackgroundColour(id);
         if ( currcolor != *wxWHITE ) {
            treectrl->SetItemBackgroundColour(id, *wxWHITE);
         }
      }
      id = treectrl->GetNextChild(root, cookie);
   }
}

// -----------------------------------------------------------------------------

// Define a window for right pane of split window:

class RightWindow : public wxWindow
{
public:
   RightWindow(wxWindow* parent, wxCoord xorg, wxCoord yorg, int wd, int ht)
      : wxWindow(parent, wxID_ANY, wxPoint(xorg,yorg), wxSize(wd,ht),
                 wxNO_BORDER |
                 // need this to avoid layer/edit/timeline bar buttons flashing on Windows
                 wxNO_FULL_REPAINT_ON_RESIZE)
   {
      // avoid erasing background on GTK+
      SetBackgroundStyle(wxBG_STYLE_CUSTOM);
   }
   ~RightWindow() {}

   // event handlers
   void OnSize(wxSizeEvent& event);
   void OnEraseBackground(wxEraseEvent& event);

   DECLARE_EVENT_TABLE()
};

BEGIN_EVENT_TABLE(RightWindow, wxWindow)
   EVT_SIZE             (RightWindow::OnSize)
   EVT_ERASE_BACKGROUND (RightWindow::OnEraseBackground)
END_EVENT_TABLE()

// -----------------------------------------------------------------------------

void RightWindow::OnSize(wxSizeEvent& event)
{
   int wd, ht;
   GetClientSize(&wd, &ht);
   if (wd > 0 && ht > 0 && bigview) {
      // resize layer/edit/timeline bars and main viewport window
      ResizeLayerBar(wd);
      ResizeEditBar(wd);
      int y = 0;
      if (showlayer) {
         y += LayerBarHeight();
         ht -= LayerBarHeight();
      }
      if (showedit) {
         y += EditBarHeight();
         ht -= EditBarHeight();
      }
      if (showtimeline) {
         ht -= TimelineBarHeight();
      }
      // timeline bar goes underneath viewport
      ResizeTimelineBar(y + ht, wd);
      bigview->SetSize(0, y, wd, ht);
   }
   event.Skip();
}

// -----------------------------------------------------------------------------

void RightWindow::OnEraseBackground(wxEraseEvent& WXUNUSED(event))
{
   // do nothing because layer/edit/timeline bars and viewport cover all of right pane
}

// -----------------------------------------------------------------------------

RightWindow* rightpane;

wxWindow* MainFrame::RightPane()
{
   return rightpane;
}

// -----------------------------------------------------------------------------

void MainFrame::ResizeSplitWindow(int wd, int ht)
{
   int x = showtool ? toolbarwd : 0;
   int y = statusptr->statusht;
   int w = showtool ? wd - toolbarwd : wd;
   int h = ht > statusptr->statusht ? ht - statusptr->statusht : 0;
   splitwin->SetSize(x, y, w, h);

#ifdef __WXOSX__
   // need this call to resize left and right panes
   splitwin->UpdateSize();
#else
   // wxSplitterWindow automatically resizes left and right panes;
   // note that RightWindow::OnSize has now been called
#endif
}

// -----------------------------------------------------------------------------

void MainFrame::ResizeStatusBar(int wd, int ht)
{
   wxUnusedVar(ht);
   // assume showstatus is true
   statusptr->statusht = showexact ? STATUS_EXHT : STATUS_HT;
   statusptr->SetSize(showtool ? toolbarwd : 0, 0,
                      showtool ? wd - toolbarwd : wd, statusptr->statusht);
}

// -----------------------------------------------------------------------------

void MainFrame::ToggleStatusBar()
{
   int wd, ht;
   GetClientSize(&wd, &ht);
   showstatus = !showstatus;
   if (showstatus) {
      ResizeStatusBar(wd, ht);
   } else {
      statusptr->statusht = 0;
      statusptr->SetSize(0, 0, 0, 0);
   }
   ResizeSplitWindow(wd, ht);
   UpdateEverything();
}

// -----------------------------------------------------------------------------

void MainFrame::ToggleExactNumbers()
{
   int wd, ht;
   GetClientSize(&wd, &ht);
   showexact = !showexact;
   if (showstatus) {
      ResizeStatusBar(wd, ht);
      ResizeSplitWindow(wd, ht);
      UpdateEverything();
   } else if (showexact) {
      // show the status bar using new size
      ToggleStatusBar();
   } else {
      UpdateMenuItems(IsActive());
   }
}

// -----------------------------------------------------------------------------

void MainFrame::ToggleToolBar()
{
   showtool = !showtool;
   int wd, ht;
   GetClientSize(&wd, &ht);
   if (showstatus) {
      ResizeStatusBar(wd, ht);
   }
   if (showtool) {
      // resize tool bar in case window was made larger while tool bar hidden
      toolbarptr->SetSize(0, 0, toolbarwd, ht);
   }
   ResizeSplitWindow(wd, ht);
   toolbarptr->Show(showtool);
}

// -----------------------------------------------------------------------------

void MainFrame::ToggleFullScreen()
{
   static bool restorestatusbar;    // restore status bar at end of full screen mode?
   static bool restorelayerbar;     // restore layer bar?
   static bool restoreeditbar;      // restore edit bar?
   static bool restoretimelinebar;  // restore timeline bar?
   static bool restoretoolbar;      // restore tool bar?
   static bool restorepattdir;      // restore pattern directory?
   static bool restorescrdir;       // restore script directory?

   if (!fullscreen) {
      // save current location and size for use in SavePrefs
      wxRect r = GetRect();
      mainx = r.x;
      mainy = r.y;
      mainwd = r.width;
      mainht = r.height;
   }

   fullscreen = !fullscreen;
   ShowFullScreen(fullscreen,
      wxFULLSCREEN_NOMENUBAR | wxFULLSCREEN_NOBORDER | wxFULLSCREEN_NOCAPTION);

   if (fullscreen) {
      // hide scroll bars
      bigview->SetScrollbar(wxHORIZONTAL, 0, 0, 0, true);
      bigview->SetScrollbar(wxVERTICAL, 0, 0, 0, true);
      
      // hide status bar if necessary
      restorestatusbar = showstatus;
      if (restorestatusbar) {
         showstatus = false;
         statusptr->statusht = 0;
         statusptr->SetSize(0, 0, 0, 0);
      }
      
      // hide layer bar if necessary
      restorelayerbar = showlayer;
      if (restorelayerbar) {
         ToggleLayerBar();
      }
      
      // hide edit bar if necessary
      restoreeditbar = showedit;
      if (restoreeditbar) {
         ToggleEditBar();
      }
      
      // hide timeline bar if necessary
      restoretimelinebar = showtimeline;
      if (restoretimelinebar) {
         ToggleTimelineBar();
      }
      
      // hide tool bar if necessary
      restoretoolbar = showtool;
      if (restoretoolbar) {
         ToggleToolBar();
      }
      
      // hide pattern/script directory if necessary
      restorepattdir = showpatterns;
      restorescrdir = showscripts;
      if (restorepattdir) {
         dirwinwd = splitwin->GetSashPosition();
         splitwin->Unsplit(patternctrl);
         showpatterns = false;
      } else if (restorescrdir) {
         dirwinwd = splitwin->GetSashPosition();
         splitwin->Unsplit(scriptctrl);
         showscripts = false;
      }

   } else {
      // first show tool bar if necessary
      if (restoretoolbar && !showtool) {
         ToggleToolBar();
         if (showstatus) {
            // reduce width of status bar below
            restorestatusbar = true;
         }
      }
      
      // show status bar if necessary;
      // note that even if it's visible we may have to resize width
      if (restorestatusbar) {
         showstatus = true;
         int wd, ht;
         GetClientSize(&wd, &ht);
         ResizeStatusBar(wd, ht);
      }

      // show layer bar if necessary
      if (restorelayerbar && !showlayer) ToggleLayerBar();

      // show edit bar if necessary
      if (restoreeditbar && !showedit) ToggleEditBar();

      // show timeline bar if necessary
      if (restoretimelinebar && !showtimeline) ToggleTimelineBar();

      // restore pattern/script directory if necessary
      if ( restorepattdir && !splitwin->IsSplit() ) {
         splitwin->SplitVertically(patternctrl, RightPane(), dirwinwd);
         showpatterns = true;
      } else if ( restorescrdir && !splitwin->IsSplit() ) {
         splitwin->SplitVertically(scriptctrl, RightPane(), dirwinwd);
         showscripts = true;
      }
   }

   if (!fullscreen) {
      // restore scroll bars BEFORE setting viewport size
      bigview->UpdateScrollBars();
   }
   
   // adjust size of viewport (and pattern/script directory if visible)
   int wd, ht;
   GetClientSize(&wd, &ht);
   ResizeSplitWindow(wd, ht);
   UpdateEverything();
}

// -----------------------------------------------------------------------------

void MainFrame::ToggleAllowUndo()
{
   if (generating) {
      // terminate generating loop and set command_pending flag
      Stop();
      command_pending = true;
      cmdevent.SetId(ID_NO_UNDO);
      return;
   }

   allowundo = !allowundo;
   if (allowundo) {
      if (currlayer->algo->getGeneration() > currlayer->startgen) {
         // undo list is empty but user can Reset, so add a generating change
         // to undo list so user can Undo or Reset (and then Redo if they wish)
         currlayer->undoredo->AddGenChange();
      }
   } else {
      currlayer->undoredo->ClearUndoRedo();
      // don't clear undo/redo history for other layers here; only do it
      // if allowundo is false when user switches to another layer
   }
}

// -----------------------------------------------------------------------------

void MainFrame::ShowPatternInfo()
{
   if (viewptr->waitingforclick || currlayer->currfile.IsEmpty()) return;
   ShowInfo(currlayer->currfile);
}

// -----------------------------------------------------------------------------

// event table and handlers for main window:

BEGIN_EVENT_TABLE(MainFrame, wxFrame)
   EVT_MENU                (wxID_ANY,        MainFrame::OnMenu)
   EVT_SET_FOCUS           (                 MainFrame::OnSetFocus)
   EVT_ACTIVATE            (                 MainFrame::OnActivate)
   EVT_IDLE                (                 MainFrame::OnIdle)
   EVT_SIZE                (                 MainFrame::OnSize)
#if defined(__WXMAC__) || defined(__WXMSW__)
   EVT_TREE_ITEM_EXPANDED   (wxID_TREECTRL,  MainFrame::OnDirTreeExpand)
   EVT_TREE_ITEM_COLLAPSING (wxID_TREECTRL,  MainFrame::OnDirTreeCollapse)
   // wxMac bug??? EVT_TREE_ITEM_COLLAPSED doesn't get called
#endif
   EVT_TREE_SEL_CHANGED    (wxID_TREECTRL,   MainFrame::OnDirTreeSelection)
   EVT_SPLITTER_DCLICK     (wxID_ANY,        MainFrame::OnSashDblClick)
   EVT_TIMER               (wxID_ANY,        MainFrame::OnOneTimer)
#if defined(__WXMAC__) || defined(__WXGTK__)
   //!!! these handlers avoid the viewport window being erased on Mac and Linux
   /* but unfortunately the split-window sash won't be drawn correctly
   EVT_ERASE_BACKGROUND    (                 MainFrame::OnErase)
   EVT_PAINT               (                 MainFrame::OnPaint)
   */
#endif
   EVT_CLOSE               (                 MainFrame::OnClose)
END_EVENT_TABLE()

// -----------------------------------------------------------------------------

void MainFrame::OnMenu(wxCommandEvent& event)
{
   showbanner = false;
   if (keepmessage) {
      // don't clear message created by script while generating a pattern
      keepmessage = false;
   } else {
      statusptr->ClearMessage();
   }
   
   int id = event.GetId();
   switch (id) {
      // File menu
      case wxID_NEW:          NewPattern(); break;
      case wxID_OPEN:         OpenPattern(); break;
      case ID_OPEN_CLIP:      OpenClipboard(); break;
      case ID_SHOW_PATTERNS:  ToggleShowPatterns(); break;
      case ID_PATTERN_DIR:    ChangePatternDir(); break;
      case wxID_SAVE:         SavePattern(); break;
      case ID_SAVE_XRLE:      savexrle = !savexrle; break;
      case ID_RUN_SCRIPT:     OpenScript(); break;
      case ID_RUN_CLIP:       RunClipboard(); break;
      case ID_SHOW_SCRIPTS:   ToggleShowScripts(); break;
      case ID_SCRIPT_DIR:     ChangeScriptDir(); break;
      case wxID_PREFERENCES:  ShowPrefsDialog(); break;
      case wxID_EXIT:         QuitApp(); break;

      // Edit menu
      case ID_UNDO:           currlayer->undoredo->UndoChange(); break;
      case ID_REDO:           currlayer->undoredo->RedoChange(); break;
      case ID_NO_UNDO:        ToggleAllowUndo(); break;
      case ID_CUT:            viewptr->CutSelection(); break;
      case ID_COPY:           viewptr->CopySelection(); break;
      case ID_CLEAR:          viewptr->ClearSelection(); break;
      case ID_OUTSIDE:        viewptr->ClearOutsideSelection(); break;
      case ID_PASTE:          viewptr->PasteClipboard(false); break;
      case ID_PASTE_SEL:      viewptr->PasteClipboard(true); break;
      case ID_PL_TL:          SetPasteLocation("TopLeft"); break;
      case ID_PL_TR:          SetPasteLocation("TopRight"); break;
      case ID_PL_BR:          SetPasteLocation("BottomRight"); break;
      case ID_PL_BL:          SetPasteLocation("BottomLeft"); break;
      case ID_PL_MID:         SetPasteLocation("Middle"); break;
      case ID_PM_AND:         SetPasteMode("And"); break;
      case ID_PM_COPY:        SetPasteMode("Copy"); break;
      case ID_PM_OR:          SetPasteMode("Or"); break;
      case ID_PM_XOR:         SetPasteMode("Xor"); break;
      case ID_SELECTALL:      viewptr->SelectAll(); break;
      case ID_REMOVE:         viewptr->RemoveSelection(); break;
      case ID_SHRINK:         viewptr->ShrinkSelection(false); break;
      case ID_SHRINKFIT:      viewptr->ShrinkSelection(true); break;
      case ID_RANDOM:         viewptr->RandomFill(); break;
      case ID_FLIPTB:         viewptr->FlipSelection(true); break;
      case ID_FLIPLR:         viewptr->FlipSelection(false); break;
      case ID_ROTATEC:        viewptr->RotateSelection(true); break;
      case ID_ROTATEA:        viewptr->RotateSelection(false); break;
      case ID_DRAW:           viewptr->SetCursorMode(curs_pencil); break;
      case ID_PICK:           viewptr->SetCursorMode(curs_pick); break;
      case ID_SELECT:         viewptr->SetCursorMode(curs_cross); break;
      case ID_MOVE:           viewptr->SetCursorMode(curs_hand); break;
      case ID_ZOOMIN:         viewptr->SetCursorMode(curs_zoomin); break;
      case ID_ZOOMOUT:        viewptr->SetCursorMode(curs_zoomout); break;

      // Control menu
      case ID_START:          if (inscript || generating) {
                                 Stop();
                              } else if (TimelineExists()) {
                                 PlayTimeline(1);
                              } else {
                                 GeneratePattern();
                              }
                              break;
      case ID_NEXT:           NextGeneration(false); break;
      case ID_STEP:           NextGeneration(true); break;
      case ID_RESET:          ResetPattern(); break;
      case ID_SETGEN:         SetGeneration(); break;
      case ID_FASTER:         GoFaster(); break;
      case ID_SLOWER:         GoSlower(); break;
      case ID_SETBASE:        SetBaseStep(); break;
      case ID_AUTO:           ToggleAutoFit(); break;
      case ID_HYPER:          ToggleHyperspeed(); break;
      case ID_HINFO:          ToggleHashInfo(); break;
      case ID_RECORD:         StartStopRecording(); break;
      case ID_DELTIME:        DeleteTimeline(); break;
      case ID_SETRULE:        ShowRuleDialog(); break;

      // View menu
      case ID_FULL:           ToggleFullScreen(); break;
      case ID_FIT:            viewptr->FitPattern(); break;
      case ID_FIT_SEL:        viewptr->FitSelection(); break;
      case ID_MIDDLE:         viewptr->ViewOrigin(); break;
      case ID_RESTORE00:      viewptr->RestoreOrigin(); break;
      case wxID_ZOOM_IN:      viewptr->ZoomIn(); break;
      case wxID_ZOOM_OUT:     viewptr->ZoomOut(); break;
      case ID_SCALE_1:        viewptr->SetPixelsPerCell(1); break;
      case ID_SCALE_2:        viewptr->SetPixelsPerCell(2); break;
      case ID_SCALE_4:        viewptr->SetPixelsPerCell(4); break;
      case ID_SCALE_8:        viewptr->SetPixelsPerCell(8); break;
      case ID_SCALE_16:       viewptr->SetPixelsPerCell(16); break;
      case ID_TOOL_BAR:       ToggleToolBar(); break;
      case ID_LAYER_BAR:      ToggleLayerBar(); break;
      case ID_EDIT_BAR:       ToggleEditBar(); break;
      case ID_ALL_STATES:     ToggleAllStates(); break;
      case ID_STATUS_BAR:     ToggleStatusBar(); break;
      case ID_EXACT:          ToggleExactNumbers(); break;
      case ID_GRID:           viewptr->ToggleGridLines(); break;
      case ID_ICONS:          viewptr->ToggleCellIcons(); break;
      case ID_INVERT:         viewptr->ToggleCellColors(); break;
      case ID_BUFF:           viewptr->ToggleBuffering(); break;
      case ID_TIMELINE:       ToggleTimelineBar(); break;
      case ID_INFO:           ShowPatternInfo(); break;

      // Layer menu
      case ID_ADD_LAYER:      AddLayer(); break;
      case ID_CLONE:          CloneLayer(); break;
      case ID_DUPLICATE:      DuplicateLayer(); break;
      case ID_DEL_LAYER:      DeleteLayer(); break;
      case ID_DEL_OTHERS:     DeleteOtherLayers(); break;
      case ID_MOVE_LAYER:     MoveLayerDialog(); break;
      case ID_NAME_LAYER:     NameLayerDialog(); break;
      case ID_SET_COLORS:     SetLayerColors(); break;
      case ID_SYNC_VIEW:      ToggleSyncViews(); break;
      case ID_SYNC_CURS:      ToggleSyncCursors(); break;
      case ID_STACK:          ToggleStackLayers(); break;
      case ID_TILE:           ToggleTileLayers(); break;

      // Help menu
      case ID_HELP_INDEX:     ShowHelp(_("Help/index.html")); break;
      case ID_HELP_INTRO:     ShowHelp(_("Help/intro.html")); break;
      case ID_HELP_TIPS:      ShowHelp(_("Help/tips.html")); break;
      case ID_HELP_ALGOS:     ShowHelp(_("Help/algos.html")); break;
      case ID_HELP_LEXICON:   ShowHelp(_("Help/Lexicon/lex.htm")); break;
      case ID_HELP_ARCHIVES:  ShowHelp(_("Help/archives.html")); break;
      case ID_HELP_PERL:      ShowHelp(_("Help/perl.html")); break;
      case ID_HELP_PYTHON:    ShowHelp(_("Help/python.html")); break;
      case ID_HELP_KEYBOARD:  ShowHelp(SHOW_KEYBOARD_SHORTCUTS); break;
      case ID_HELP_MOUSE:     ShowHelp(_("Help/mouse.html")); break;
      case ID_HELP_FILE:      ShowHelp(_("Help/file.html")); break;
      case ID_HELP_EDIT:      ShowHelp(_("Help/edit.html")); break;
      case ID_HELP_CONTROL:   ShowHelp(_("Help/control.html")); break;
      case ID_HELP_VIEW:      ShowHelp(_("Help/view.html")); break;
      case ID_HELP_LAYER:     ShowHelp(_("Help/layer.html")); break;
      case ID_HELP_HELP:      ShowHelp(_("Help/help.html")); break;
      case ID_HELP_REFS:      ShowHelp(_("Help/refs.html")); break;
      case ID_HELP_FORMATS:   ShowHelp(_("Help/formats.html")); break;
      case ID_HELP_BOUNDED:   ShowHelp(_("Help/bounded.html")); break;
      case ID_HELP_PROBLEMS:  ShowHelp(_("Help/problems.html")); break;
      case ID_HELP_CHANGES:   ShowHelp(_("Help/changes.html")); break;
      case ID_HELP_CREDITS:   ShowHelp(_("Help/credits.html")); break;
      case ID_HELP_BUTT:      ShowHelp(wxEmptyString); break;
      case wxID_ABOUT:        ShowAboutBox(); break;

      // Open/Run Recent submenus
      case ID_CLEAR_MISSING_PATTERNS:  ClearMissingPatterns(); break;
      case ID_CLEAR_ALL_PATTERNS:      ClearAllPatterns(); break;
      case ID_CLEAR_MISSING_SCRIPTS:   ClearMissingScripts(); break;
      case ID_CLEAR_ALL_SCRIPTS:       ClearAllScripts(); break;

      default:
         if ( id > ID_OPEN_RECENT && id <= ID_OPEN_RECENT + numpatterns ) {
            OpenRecentPattern(id);

         } else if ( id > ID_RUN_RECENT && id <= ID_RUN_RECENT + numscripts ) {
            OpenRecentScript(id);

         } else if ( id >= ID_ALGO0 && id <= ID_ALGOMAX ) {
            int newtype = id - ID_ALGO0;
            ChangeAlgorithm(newtype);

         } else if ( id >= ID_LAYER0 && id <= ID_LAYERMAX ) {
            SetLayer(id - ID_LAYER0);

         } else {
            // wxOSX app needs this to handle app menu items like "Hide Golly"
            event.Skip();
         }
   }
   
   UpdateUserInterface(IsActive());

   if (inscript) {
      // update viewport, status bar, scroll bars, window title
      inscript = false;
      UpdatePatternAndStatus();
      bigview->UpdateScrollBars();
      SetWindowTitle(wxEmptyString);
      inscript = true;
   }
}

// -----------------------------------------------------------------------------

void MainFrame::OnSetFocus(wxFocusEvent& WXUNUSED(event))
{
   // this is never called in Mac app, presumably because it doesn't
   // make any sense for a wxFrame to get the keyboard focus

   #ifdef __WXMSW__
      // fix wxMSW problem: don't let main window get focus after being minimized
      if (viewptr) viewptr->SetFocus();
   #endif
}

// -----------------------------------------------------------------------------

void MainFrame::OnActivate(wxActivateEvent& event)
{
   // note that IsActive() doesn't always match event.GetActive()

   if (viewptr->oldcursor != NULL && !event.GetActive()) {
      // this can happen if the shift key is used in a keyboard shortcut that
      // opens a dialog or window, which means PatternView::OnKeyUp isn't called
      viewptr->SetCursorMode(viewptr->oldcursor);
      viewptr->oldcursor = NULL;
      UpdateEditBar(false);
   }

   #if defined(__WXMAC__) && !defined(__WXOSX_COCOA__)
      // to avoid disabled menu items after a modal dialog closes
      // don't call UpdateMenuItems on deactivation
      if (event.GetActive()) {
         UpdateUserInterface(true);
         // need to set focus to avoid next IsActive() call returning false
         // (presumably due to a wxMac bug)
         viewptr->SetFocus();
      } else {
         wxSetCursor(*wxSTANDARD_CURSOR);
      }
   #else
      UpdateUserInterface(event.GetActive());
   #endif

   #if defined(__WXGTK__) && !wxCHECK_VERSION(2,9,0)
      /* wxGTK 2.8 requires this hack to re-enable menu items after a modal
         dialog closes.  With wxGTK 2.9 is causes deadlocks due to concurrent
         calls to UpdateMenuItems() (which calls ClipboardHasText() which is
         non-reentrant).  Maybe caused by timer events not being dispatched
         from the GUI thread?  */
      if (event.GetActive()) onetimer->Start(20, wxTIMER_ONE_SHOT);
      // OnOneTimer will be called after delay of 0.02 secs
   #endif
   
   event.Skip();
}

// -----------------------------------------------------------------------------

void MainFrame::OnSize(wxSizeEvent& event)
{
   #ifdef __WXMSW__
      // save current location and size for use in SavePrefs if app
      // is closed when window is minimized
      wxRect r = GetRect();
      mainx = r.x;
      mainy = r.y;
      mainwd = r.width;
      mainht = r.height;
   #endif

   int wd, ht;
   GetClientSize(&wd, &ht);
   if (wd > 0 && ht > 0) {
      // toolbarptr/statusptr/viewptr might be NULL if OnSize is called from MainFrame ctor
      if (toolbarptr && showtool) {
         // adjust size of tool bar
         toolbarptr->SetSize(0, 0, toolbarwd, ht);
      }
      if (statusptr && showstatus) {
         // adjust size of status bar
         ResizeStatusBar(wd, ht);
      }
      if (viewptr && statusptr && ht > statusptr->statusht) {
         // adjust size of viewport (and pattern/script directory if visible)
         ResizeSplitWindow(wd, ht);
      }
   }
   
   #ifdef __WXGTK__
      // need to do default processing for menu bar and tool bar
      event.Skip();
   #else
      wxUnusedVar(event);
   #endif
}

// -----------------------------------------------------------------------------

// avoid recursive call of OpenFile in OnIdle;
// this can happen if user clicks a script which then opens some sort of dialog
// (idle events are sent to the main window while the dialog is open)
static bool inidle = false;

void MainFrame::OnIdle(wxIdleEvent& event)
{
   if (inidle) return;
   
   #ifdef __WXMSW__
      if ( call_unselect ) {
         // deselect file/folder so user can click the same item
         if (showpatterns) patternctrl->GetTreeCtrl()->Unselect();
         if (showscripts) scriptctrl->GetTreeCtrl()->Unselect();
         call_unselect = false;
         
         // calling SetFocus once doesn't stuff up layer bar buttons
         if ( IsActive() && viewptr ) viewptr->SetFocus();
      }
      
      if (!editpath.IsEmpty()) {
         EditFile(editpath);
         editpath.Clear();
      }
   #else
      // ensure viewport window has keyboard focus if main window is active;
      // note that we can't do this on Windows because it stuffs up clicks
      // in layer bar buttons
      if ( IsActive() && viewptr ) viewptr->SetFocus();
   #endif

   // process any pending script/pattern files
   if ( pendingfiles.GetCount() > 0 ) {
      inidle = true;
      for ( size_t n = 0; n < pendingfiles.GetCount(); n++ ) {
         OpenFile(pendingfiles[n]);
      }
      inidle = false;
      pendingfiles.Clear();
   }
  
   if (call_close) {
      call_close = false;
      Close(false);        // false allows OnClose handler to veto close
   }
   
   if (TimelineExists() && AutoPlay()) {
      // in autoplay mode so we need another idle event
      // (this works much better than calling wxWakeUpIdle)
      event.RequestMore();
   }
   
   event.Skip();
}

// -----------------------------------------------------------------------------

void MainFrame::OnDirTreeExpand(wxTreeEvent& WXUNUSED(event))
{
   #ifdef __WXMAC__
      if ((generating || inscript) && (showpatterns || showscripts)) {
         // send idle event so directory tree gets updated
         #if wxCHECK_VERSION(2,9,2)
            // SendIdleEvents is no lnger a member of wxApp -- need to fix???!!!
         #else
            wxIdleEvent idleevent;
            wxGetApp().SendIdleEvents(this, idleevent);
         #endif
      }
   #endif
   #ifdef __WXMSW__
      // avoid bug -- expanding/collapsing a folder can cause top visible item
      // to become selected and thus OnDirTreeSelection gets called
      ignore_selection = true;
   #endif
}

// -----------------------------------------------------------------------------

void MainFrame::OnDirTreeCollapse(wxTreeEvent& WXUNUSED(event))
{
   #ifdef __WXMAC__
      if ((generating || inscript) && (showpatterns || showscripts)) {
         // send idle event so directory tree gets updated
         #if wxCHECK_VERSION(2,9,2)
            // SendIdleEvents is no lnger a member of wxApp -- need to fix???!!!
         #else
            wxIdleEvent idleevent;
            wxGetApp().SendIdleEvents(this, idleevent);
         #endif
      }
   #endif
   #ifdef __WXMSW__
      // avoid bug -- expanding/collapsing a folder can cause top visible item
      // to become selected and thus OnDirTreeSelection gets called
      ignore_selection = true;
   #endif
}

// -----------------------------------------------------------------------------

#if defined(__WXOSX__) || defined(__WXCOCOA__)
   // wxMOD_CONTROL has been changed to mean Command key down (sheesh!)
   #define wxMOD_CONTROL wxMOD_RAW_CONTROL
   #define ControlDown RawControlDown
#endif

void MainFrame::OnTreeClick(wxMouseEvent& event)
{
   // set global flag for testing in OnDirTreeSelection
   edit_file = event.ControlDown() || event.RightDown();
   
#ifdef __WXMSW__
   // this handler gets called even if user clicks outside an item,
   // and in some cases can result in the top visible item becoming
   // selected, so we need to avoid that
   ignore_selection = false;
   wxGenericDirCtrl* dirctrl = NULL;
   // for some reason we need to use mainptr to access next 2 members
   // (it's something to do with using Connect)
   if (showpatterns) dirctrl = mainptr->patternctrl;
   if (showscripts) dirctrl = mainptr->scriptctrl;
   if (dirctrl) {
      wxTreeCtrl* treectrl = dirctrl->GetTreeCtrl();
      if (treectrl) {
         wxPoint pt = event.GetPosition();
         int flags;
         wxTreeItemId id = treectrl->HitTest(pt, flags);
         if (id.IsOk() && (flags & wxTREE_HITTEST_ONITEMLABEL ||
                           flags & wxTREE_HITTEST_ONITEMICON)) {
            // fix problem with right-click
            if (event.RightDown()) {
               treectrl->SelectItem(id, true);
               // OnDirTreeSelection gets called a few times for some reason
            }
            // fix problem with double-click
            if (event.LeftDClick()) ignore_selection = true;
         } else {
            ignore_selection = true;
         }
      }
   }
#endif
   
   event.Skip();
}

// -----------------------------------------------------------------------------

void MainFrame::EditFile(const wxString& filepath)
{
   // prompt user if text editor hasn't been set yet
   if (texteditor.IsEmpty()) {
      ChooseTextEditor(this, texteditor);
      if (texteditor.IsEmpty()) return;
   }
   
   // open given pattern/script file in user's preferred text editor
   wxString cmd;
   #ifdef __WXMAC__
      cmd = wxString::Format(wxT("open -a \"%s\" \"%s\""), texteditor.c_str(), filepath.c_str());
   #else
      // Windows or Unix
      cmd = wxString::Format(wxT("\"%s\" \"%s\""), texteditor.c_str(), filepath.c_str());
   #endif
   wxExecute(cmd, wxEXEC_ASYNC);
}

// -----------------------------------------------------------------------------

void MainFrame::OnDirTreeSelection(wxTreeEvent& event)
{
   // note that viewptr will be NULL if called from MainFrame::MainFrame
   if ( viewptr ) {
      wxTreeItemId id = event.GetItem();
      if ( !id.IsOk() ) return;

      wxGenericDirCtrl* dirctrl = NULL;
      if (showpatterns) dirctrl = patternctrl;
      if (showscripts) dirctrl = scriptctrl;
      if (dirctrl == NULL) return;
      
      wxString filepath = dirctrl->GetFilePath();

      // deselect file/folder so this handler will be called if user clicks same item
      wxTreeCtrl* treectrl = dirctrl->GetTreeCtrl();
      #ifdef __WXMSW__
         // calling UnselectAll() or Unselect() here causes a crash
         if (ignore_selection) {
            // ignore spurious selection
            ignore_selection = false;
            call_unselect = true;
            return;
         }
      #else
         treectrl->UnselectAll();
      #endif

      if ( filepath.IsEmpty() ) {
         // user clicked on a folder name so expand or collapse it???
         // unfortunately, using Collapse/Expand causes this handler to be
         // called again and there's no easy way to distinguish between
         // a click in the folder name or a dbl-click (or a click in the
         // +/-/arrow image)
         /*
         if ( treectrl->IsExpanded(id) ) {
            treectrl->Collapse(id);
         } else {
            treectrl->Expand(id);
         }
         */

      } else if (edit_file) {
         // open file in text editor
         #ifdef __WXMSW__
            // call EditFile in later OnIdle to avoid right-click problem
            editpath = filepath;
         #else
            EditFile(filepath);
         #endif

      } else {
         // user clicked on a file name
         if ( inscript ) {
            // use Warning because statusptr->ErrorMessage does nothing if inscript
            if ( IsScriptFile(filepath) )
               Warning(_("Cannot run script while another script is running."));
            else
               Warning(_("Cannot open file while a script is running."));
         } else {
            // reset background of previously selected file by traversing entire tree;
            // we can't just remember previously selected id because ids don't persist
            // after a folder has been collapsed and expanded
            DeselectTree(treectrl, treectrl->GetRootItem());

            // indicate the selected file
            treectrl->SetItemBackgroundColour(id, *wxLIGHT_GREY);

            #ifdef __WXMAC__
               if ( !wxFileName::FileExists(filepath) ) {
                  // avoid wxMac bug in wxGenericDirCtrl::GetFilePath; ie. file name
                  // can contain "/" rather than ":" (but directory path is okay)
                  wxFileName fullpath(filepath);
                  wxString dir = fullpath.GetPath();
                  wxString name = fullpath.GetFullName();
                  wxString newpath = dir + wxT(":") + name;
                  if ( wxFileName::FileExists(newpath) ) filepath = newpath;
               }
            #endif

            if (generating) {
               // terminate generating loop and set command_pending flag
               Stop();
               command_pending = true;
               if ( IsScriptFile(filepath) ) {
                  AddRecentScript(filepath);
                  cmdevent.SetId(ID_RUN_RECENT + 1);
               } else {
                  AddRecentPattern(filepath);
                  cmdevent.SetId(ID_OPEN_RECENT + 1);
               }
            } else {
               // load pattern or run script
               // OpenFile(filepath);
               // call OpenFile in later OnIdle -- this prevents the main window
               // moving in front of the help window if a script calls help(...)
               pendingfiles.Add(filepath);
            }
         }
      }

      #ifdef __WXMSW__
         // calling Unselect() here causes a crash so do later in OnIdle
         call_unselect = true;
      #endif

      // changing focus here doesn't work in Mac or Win apps
      // (presumably because they set focus to treectrl after this call)
      viewptr->SetFocus();
   }
}

// -----------------------------------------------------------------------------

void MainFrame::OnSashDblClick(wxSplitterEvent& WXUNUSED(event))
{
   // splitwin's sash was double-clicked
   if (showpatterns) ToggleShowPatterns();
   if (showscripts) ToggleShowScripts();
   UpdateMenuItems(IsActive());
   UpdateToolBar(IsActive());
}

// -----------------------------------------------------------------------------

void MainFrame::OnOneTimer(wxTimerEvent& WXUNUSED(event))
{
   // fix drag and drop problem on Mac -- see DnDFile::OnDropFiles
   #ifdef __WXMAC__
      // remove colored frame
      if (viewptr) RefreshView();
   #endif
   
   // fix menu item problem on Linux after modal dialog has closed
   #ifdef __WXGTK__
      UpdateMenuItems(true);
   #endif
}

// -----------------------------------------------------------------------------

bool MainFrame::SaveCurrentLayer()
{
   if (currlayer->algo->isEmpty()) return true;    // no need to save empty universe
   wxString query;
   if (numlayers > 1) {
      // make it clear which layer we're asking about
      query.Printf(_("Save the changes to layer %d: \"%s\"?"),
                   currindex, currlayer->currname.c_str());
   } else {
      query.Printf(_("Save the changes to \"%s\"?"), currlayer->currname.c_str());
   }
   int answer = SaveChanges(query, _("If you don't save, your changes will be lost."));
   switch (answer) {
      case 2:  return SavePattern();   // true only if pattern was saved
      case 1:  return true;            // don't save changes
      default: return false;           // answer == 0 (ie. user selected Cancel)
   }
}

// -----------------------------------------------------------------------------

void MainFrame::OnErase(wxEraseEvent& WXUNUSED(event))
{
   // do nothing
}

// -----------------------------------------------------------------------------

void MainFrame::OnPaint(wxPaintEvent& WXUNUSED(event))
{
    wxPaintDC dc(this);
    // paint nothing (on Mac this avoids drawing the gray horizontal stripes)
}

// -----------------------------------------------------------------------------

void MainFrame::OnClose(wxCloseEvent& event)
{
   if (event.CanVeto()) {
      // we can cancel the close event if necessary
      if (inscript || generating) {
         Stop();
         /* using wxPostEvent doesn't work if we've been called from Yield:
         wxCommandEvent quitevt(wxEVT_COMMAND_MENU_SELECTED, wxID_EXIT);
         wxPostEvent(this->GetEventHandler(), quitevt);
         */
         // set flag so OnClose gets called again in next OnIdle
         call_close = true;
         event.Veto();
         return;
      }
   
      if (askonquit) {
         // keep track of which unique clones have been seen;
         // we add 1 below to allow for cloneseen[0] (always false)
         const int maxseen = MAX_LAYERS/2 + 1;
         bool cloneseen[maxseen];
         for (int i = 0; i < maxseen; i++) cloneseen[i] = false;
      
         // for each dirty layer, ask user if they want to save changes
         int oldindex = currindex;
         for (int i = 0; i < numlayers; i++) {
            // only ask once for each unique clone (cloneid == 0 for non-clone)
            int cid = GetLayer(i)->cloneid;
            if (!cloneseen[cid]) {
               if (cid > 0) cloneseen[cid] = true;
               if (GetLayer(i)->dirty) {
                  if (i != currindex) SetLayer(i);
                  if (!SaveCurrentLayer()) {
                     // user cancelled "save changes" dialog so restore layer
                     SetLayer(oldindex);
                     UpdateUserInterface(IsActive());
                     event.Veto();
                     return;
                  }
               }
            }
         }
      }
   }

   if (GetHelpFrame()) GetHelpFrame()->Close(true);
   if (GetInfoFrame()) GetInfoFrame()->Close(true);
   
   if (splitwin->IsSplit()) dirwinwd = splitwin->GetSashPosition();
   
   // if script is running or pattern is generating then CanVeto was false
   // (probably due to user logging out or shutting down system)
   // and we need to call exit below
   bool callexit = inscript || generating;

   // abort any running script and tidy up; also restores current directory
   // to location of Golly app so prefs file will be saved in correct place
   FinishScripting();

   // save main window location and other user preferences
   SavePrefs();
   
   // delete any temporary files
   if (wxFileExists(perlfile)) wxRemoveFile(perlfile);
   if (wxFileExists(pythonfile)) wxRemoveFile(pythonfile);
   for (int i = 0; i < numlayers; i++) {
      Layer* layer = GetLayer(i);
      if (wxFileExists(layer->tempstart)) wxRemoveFile(layer->tempstart);
      // clear all undo/redo history for this layer
      layer->undoredo->ClearUndoRedo();
   }
   
   // delete all files in tempdir (we assume it has no subdirs)
   wxDir* dir = new wxDir(tempdir);
   wxArrayString files;
   wxString filename;
   bool more = dir->GetFirst(&filename);
   while (more) {
      files.Add(tempdir + filename);
      more = dir->GetNext(&filename);
   }
   // delete wxDir object now otherwise Rmdir below will fail on Windows
   delete dir;
   for (size_t n = 0; n < files.GetCount(); n++) {
      wxRemoveFile(files[n]);
   }
   // delete the (hopefully) empty tempdir
   if (!wxFileName::Rmdir(tempdir)) {
      Warning(_("Could not delete temporary directory:\n") + tempdir);
   }
   
   // allow clipboard data to persist after app exits
   // (needed on Windows, not needed on Mac, doesn't work on Linux -- sheesh!)
   if (wxTheClipboard->Open()) {
      wxTheClipboard->Flush();
      wxTheClipboard->Close();
   }

   // avoid possible error message or seg fault
   if (callexit) exit(0);
   
   Destroy();
   
   #ifdef __WXGTK__
      // avoid seg fault on Linux (only happens if ctrl-Q is used to quit)
      exit(0);
   #endif
}

// -----------------------------------------------------------------------------

void MainFrame::QuitApp()
{
   if (viewptr->waitingforclick) return;

   Close(false);   // false allows OnClose handler to veto close
}

// -----------------------------------------------------------------------------

#if wxUSE_DRAG_AND_DROP

// derive a simple class for handling dropped files
class DnDFile : public wxFileDropTarget
{
public:
   DnDFile() {}
   virtual bool OnDropFiles(wxCoord x, wxCoord y, const wxArrayString& filenames);
};

bool DnDFile::OnDropFiles(wxCoord, wxCoord, const wxArrayString& filenames)
{
   // bring app to front
   #ifdef __WXMAC__
      ProcessSerialNumber process;
      if ( GetCurrentProcess(&process) == noErr )
         SetFrontProcess(&process);
   #endif
   #ifdef __WXMSW__
      SetForegroundWindow( (HWND)mainptr->GetHandle() );
   #endif
   mainptr->Raise();
   
   size_t numfiles = filenames.GetCount();
   for ( size_t n = 0; n < numfiles; n++ ) {
      mainptr->OpenFile(filenames[n]);
   }

   #ifdef __WXMAC__
      // need to call Refresh a bit later to remove colored frame on Mac
      onetimer->Start(10, wxTIMER_ONE_SHOT);
      // OnOneTimer will be called once after a delay of 0.01 secs
   #endif
   
   return true;
}

wxDropTarget* MainFrame::NewDropTarget()
{
   return new DnDFile();
}

#endif // wxUSE_DRAG_AND_DROP

// -----------------------------------------------------------------------------

void MainFrame::SetRandomFillPercentage()
{
   // update Random Fill menu item to show randomfill value
   wxMenuBar* mbar = GetMenuBar();
   if (mbar) {
      wxString randlabel;
      randlabel.Printf(_("Random Fill (%d%c)"), randomfill, '%');
      randlabel += GetAccelerator(DO_RANDFILL);
      mbar->SetLabel(ID_RANDOM, randlabel);
   }
}

// -----------------------------------------------------------------------------

void MainFrame::UpdateLayerItem(int index)
{
   // update name in given layer's menu item
   Layer* layer = GetLayer(index);
   wxMenuBar* mbar = GetMenuBar();
   if (mbar) {
      wxString label = wxEmptyString;
      
      // we no longer show index in front of name
      // label.Printf(_("%d: "), index);

      // display asterisk if pattern has been modified
      if (layer->dirty) label += wxT("*");
      
      int cid = layer->cloneid;
      while (cid > 0) {
         // display one or more "=" chars to indicate this is a cloned layer
         label += wxT("=");
         cid--;
      }
      
      label += layer->currname;
      
      // duplicate any ampersands so they appear
      label.Replace(wxT("&"), wxT("&&"));

      mbar->SetLabel(ID_LAYER0 + index, label);
      
      // also update name in corresponding layer button
      UpdateLayerButton(index, label);
   }
}

// -----------------------------------------------------------------------------

void MainFrame::AppendLayerItem()
{
   wxMenuBar* mbar = GetMenuBar();
   if (mbar) {
      wxMenu* layermenu = mbar->GetMenu( mbar->FindMenu(_("Layer")) );
      if (layermenu) {
         // no point setting the item name here because UpdateLayerItem
         // will be called very soon
         layermenu->AppendCheckItem(ID_LAYER0 + numlayers - 1, _("foo"));
      } else {
         Warning(_("Could not find Layer menu!"));
      }
   }
}

// -----------------------------------------------------------------------------

void MainFrame::RemoveLayerItem()
{
   wxMenuBar* mbar = GetMenuBar();
   if (mbar) {
      wxMenu* layermenu = mbar->GetMenu( mbar->FindMenu(_("Layer")) );
      if (layermenu) {
         layermenu->Delete(ID_LAYER0 + numlayers);
      } else {
         Warning(_("Could not find Layer menu!"));
      }
   }
}

// -----------------------------------------------------------------------------

void MainFrame::CreateMenus()
{
   wxMenu* fileMenu = new wxMenu();
   wxMenu* editMenu = new wxMenu();
   wxMenu* controlMenu = new wxMenu();
   wxMenu* viewMenu = new wxMenu();
   wxMenu* layerMenu = new wxMenu();
   wxMenu* helpMenu = new wxMenu();

   // create submenus
   wxMenu* plocSubMenu = new wxMenu();
   wxMenu* pmodeSubMenu = new wxMenu();
   wxMenu* cmodeSubMenu = new wxMenu();
   wxMenu* scaleSubMenu = new wxMenu();

   plocSubMenu->AppendCheckItem(ID_PL_TL,       _("Top Left"));
   plocSubMenu->AppendCheckItem(ID_PL_TR,       _("Top Right"));
   plocSubMenu->AppendCheckItem(ID_PL_BR,       _("Bottom Right"));
   plocSubMenu->AppendCheckItem(ID_PL_BL,       _("Bottom Left"));
   plocSubMenu->AppendCheckItem(ID_PL_MID,      _("Middle"));

   pmodeSubMenu->AppendCheckItem(ID_PM_AND,     _("And"));
   pmodeSubMenu->AppendCheckItem(ID_PM_COPY,    _("Copy"));
   pmodeSubMenu->AppendCheckItem(ID_PM_OR,      _("Or"));
   pmodeSubMenu->AppendCheckItem(ID_PM_XOR,     _("Xor"));

   cmodeSubMenu->AppendCheckItem(ID_DRAW,       _("Draw") + GetAccelerator(DO_CURSDRAW));
   cmodeSubMenu->AppendCheckItem(ID_PICK,       _("Pick") + GetAccelerator(DO_CURSPICK));
   cmodeSubMenu->AppendCheckItem(ID_SELECT,     _("Select") + GetAccelerator(DO_CURSSEL));
   cmodeSubMenu->AppendCheckItem(ID_MOVE,       _("Move") + GetAccelerator(DO_CURSMOVE));
   cmodeSubMenu->AppendCheckItem(ID_ZOOMIN,     _("Zoom In") + GetAccelerator(DO_CURSIN));
   cmodeSubMenu->AppendCheckItem(ID_ZOOMOUT,    _("Zoom Out") + GetAccelerator(DO_CURSOUT));

   scaleSubMenu->AppendCheckItem(ID_SCALE_1,    _("1:1") + GetAccelerator(DO_SCALE1));
   scaleSubMenu->AppendCheckItem(ID_SCALE_2,    _("1:2") + GetAccelerator(DO_SCALE2));
   scaleSubMenu->AppendCheckItem(ID_SCALE_4,    _("1:4") + GetAccelerator(DO_SCALE4));
   scaleSubMenu->AppendCheckItem(ID_SCALE_8,    _("1:8") + GetAccelerator(DO_SCALE8));
   scaleSubMenu->AppendCheckItem(ID_SCALE_16,   _("1:16") + GetAccelerator(DO_SCALE16));

   fileMenu->Append(wxID_NEW,                   _("New Pattern") + GetAccelerator(DO_NEWPATT));
   fileMenu->AppendSeparator();
   fileMenu->Append(wxID_OPEN,                  _("Open Pattern...") + GetAccelerator(DO_OPENPATT));
   fileMenu->Append(ID_OPEN_CLIP,               _("Open Clipboard") + GetAccelerator(DO_OPENCLIP));
   fileMenu->Append(ID_OPEN_RECENT,             _("Open Recent"), patternSubMenu);
   fileMenu->AppendSeparator();
   fileMenu->AppendCheckItem(ID_SHOW_PATTERNS,  _("Show Patterns") + GetAccelerator(DO_PATTERNS));
   fileMenu->Append(ID_PATTERN_DIR,             _("Set Pattern Folder...") + GetAccelerator(DO_PATTDIR));
   fileMenu->AppendSeparator();
   fileMenu->Append(wxID_SAVE,                  _("Save Pattern...") + GetAccelerator(DO_SAVE));
   fileMenu->AppendCheckItem(ID_SAVE_XRLE,      _("Save Extended RLE") + GetAccelerator(DO_SAVEXRLE));
   fileMenu->AppendSeparator();
   fileMenu->Append(ID_RUN_SCRIPT,              _("Run Script...") + GetAccelerator(DO_RUNSCRIPT));
   fileMenu->Append(ID_RUN_CLIP,                _("Run Clipboard") + GetAccelerator(DO_RUNCLIP));
   fileMenu->Append(ID_RUN_RECENT,              _("Run Recent"), scriptSubMenu);
   fileMenu->AppendSeparator();
   fileMenu->AppendCheckItem(ID_SHOW_SCRIPTS,   _("Show Scripts") + GetAccelerator(DO_SCRIPTS));
   fileMenu->Append(ID_SCRIPT_DIR,              _("Set Script Folder...") + GetAccelerator(DO_SCRIPTDIR));
#if !defined(__WXOSX_COCOA__)
   fileMenu->AppendSeparator();
#endif
   // on the Mac the wxID_PREFERENCES item is moved to the app menu
   fileMenu->Append(wxID_PREFERENCES,           _("Preferences...") + GetAccelerator(DO_PREFS));
#if !defined(__WXOSX_COCOA__)
   fileMenu->AppendSeparator();
#endif
   // on the Mac the wxID_EXIT item is moved to the app menu and the app name is appended to "Quit "
   fileMenu->Append(wxID_EXIT,                  _("Quit") + GetAccelerator(DO_QUIT));

   editMenu->Append(ID_UNDO,                    _("Undo") + GetAccelerator(DO_UNDO));
   editMenu->Append(ID_REDO,                    _("Redo") + GetAccelerator(DO_REDO));
   editMenu->AppendCheckItem(ID_NO_UNDO,        _("Disable Undo/Redo") + GetAccelerator(DO_DISABLE));
   editMenu->AppendSeparator();
   editMenu->Append(ID_CUT,                     _("Cut") + GetAccelerator(DO_CUT));
   editMenu->Append(ID_COPY,                    _("Copy") + GetAccelerator(DO_COPY));
   editMenu->Append(ID_CLEAR,                   _("Clear") + GetAccelerator(DO_CLEAR));
   editMenu->Append(ID_OUTSIDE,                 _("Clear Outside") + GetAccelerator(DO_CLEAROUT));
   editMenu->AppendSeparator();
   editMenu->Append(ID_PASTE,                   _("Paste") + GetAccelerator(DO_PASTE));
   editMenu->Append(ID_PMODE,                   _("Paste Mode"), pmodeSubMenu);
   editMenu->Append(ID_PLOCATION,               _("Paste Location"), plocSubMenu);
   editMenu->Append(ID_PASTE_SEL,               _("Paste to Selection") + GetAccelerator(DO_PASTESEL));
   editMenu->AppendSeparator();
   editMenu->Append(ID_SELECTALL,               _("Select All") + GetAccelerator(DO_SELALL));
   editMenu->Append(ID_REMOVE,                  _("Remove Selection") + GetAccelerator(DO_REMOVESEL));
   editMenu->Append(ID_SHRINK,                  _("Shrink Selection") + GetAccelerator(DO_SHRINK));
   // full label will be set later by SetRandomFillPercentage
   editMenu->Append(ID_RANDOM,                  _("Random Fill") + GetAccelerator(DO_RANDFILL));
   editMenu->Append(ID_FLIPTB,                  _("Flip Top-Bottom") + GetAccelerator(DO_FLIPTB));
   editMenu->Append(ID_FLIPLR,                  _("Flip Left-Right") + GetAccelerator(DO_FLIPLR));
   editMenu->Append(ID_ROTATEC,                 _("Rotate Clockwise") + GetAccelerator(DO_ROTATECW));
   editMenu->Append(ID_ROTATEA,                 _("Rotate Anticlockwise") + GetAccelerator(DO_ROTATEACW));
   editMenu->AppendSeparator();
   editMenu->Append(ID_CMODE,                   _("Cursor Mode"), cmodeSubMenu);

   controlMenu->Append(ID_START,                _("Start Generating") + GetAccelerator(DO_STARTSTOP));
   controlMenu->Append(ID_NEXT,                 _("Next Generation") + GetAccelerator(DO_NEXTGEN));
   controlMenu->Append(ID_STEP,                 _("Next Step") + GetAccelerator(DO_NEXTSTEP));
   controlMenu->AppendSeparator();
   controlMenu->Append(ID_RESET,                _("Reset") + GetAccelerator(DO_RESET));
   controlMenu->Append(ID_SETGEN,               _("Set Generation...") + GetAccelerator(DO_SETGEN));
   controlMenu->AppendSeparator();
   controlMenu->Append(ID_FASTER,               _("Faster") + GetAccelerator(DO_FASTER));
   controlMenu->Append(ID_SLOWER,               _("Slower") + GetAccelerator(DO_SLOWER));
   controlMenu->Append(ID_SETBASE,              _("Set Base Step...") + GetAccelerator(DO_SETBASE));
   controlMenu->AppendSeparator();
   controlMenu->AppendCheckItem(ID_AUTO,        _("Auto Fit") + GetAccelerator(DO_AUTOFIT));
   controlMenu->AppendCheckItem(ID_HYPER,       _("Hyperspeed") + GetAccelerator(DO_HYPER));
   controlMenu->AppendCheckItem(ID_HINFO,       _("Show Hash Info") + GetAccelerator(DO_HASHINFO));
   controlMenu->AppendSeparator();
   controlMenu->Append(ID_RECORD,               _("Start Recording") + GetAccelerator(DO_RECORD));
   controlMenu->Append(ID_DELTIME,              _("Delete Timeline") + GetAccelerator(DO_DELTIME));
   controlMenu->AppendSeparator();
   controlMenu->Append(ID_SETALGO,              _("Set Algorithm"), algomenu);
   controlMenu->Append(ID_SETRULE,              _("Set Rule...") + GetAccelerator(DO_SETRULE));

   viewMenu->Append(ID_FULL,                    _("Full Screen") + GetAccelerator(DO_FULLSCREEN));
   viewMenu->AppendSeparator();
   viewMenu->Append(ID_FIT,                     _("Fit Pattern") + GetAccelerator(DO_FIT));
   viewMenu->Append(ID_FIT_SEL,                 _("Fit Selection") + GetAccelerator(DO_FITSEL));
   viewMenu->Append(ID_MIDDLE,                  _("Middle") + GetAccelerator(DO_MIDDLE));
   viewMenu->Append(ID_RESTORE00,               _("Restore Origin") + GetAccelerator(DO_RESTORE00));
   viewMenu->AppendSeparator();
   viewMenu->Append(wxID_ZOOM_IN,               _("Zoom In") + GetAccelerator(DO_ZOOMIN));
   viewMenu->Append(wxID_ZOOM_OUT,              _("Zoom Out") + GetAccelerator(DO_ZOOMOUT));
   viewMenu->Append(ID_SET_SCALE,               _("Set Scale"), scaleSubMenu);
   viewMenu->AppendSeparator();
   viewMenu->AppendCheckItem(ID_TOOL_BAR,       _("Show Tool Bar") + GetAccelerator(DO_SHOWTOOL));
   viewMenu->AppendCheckItem(ID_LAYER_BAR,      _("Show Layer Bar") + GetAccelerator(DO_SHOWLAYER));
   viewMenu->AppendCheckItem(ID_EDIT_BAR,       _("Show Edit Bar") + GetAccelerator(DO_SHOWEDIT));
   viewMenu->AppendCheckItem(ID_ALL_STATES,     _("Show All States") + GetAccelerator(DO_SHOWSTATES));
   viewMenu->AppendCheckItem(ID_STATUS_BAR,     _("Show Status Bar") + GetAccelerator(DO_SHOWSTATUS));
   viewMenu->AppendCheckItem(ID_EXACT,          _("Show Exact Numbers") + GetAccelerator(DO_SHOWEXACT));
   viewMenu->AppendCheckItem(ID_GRID,           _("Show Grid Lines") + GetAccelerator(DO_SHOWGRID));
   viewMenu->AppendCheckItem(ID_ICONS,          _("Show Cell Icons") + GetAccelerator(DO_SHOWICONS));
   viewMenu->AppendCheckItem(ID_INVERT,         _("Invert Colors") + GetAccelerator(DO_INVERT));
   viewMenu->AppendCheckItem(ID_BUFF,           _("Buffered") + GetAccelerator(DO_BUFFERED));
   viewMenu->AppendCheckItem(ID_TIMELINE,       _("Show Timeline") + GetAccelerator(DO_SHOWTIME));
   viewMenu->AppendSeparator();
   viewMenu->Append(ID_INFO,                    _("Pattern Info") + GetAccelerator(DO_INFO));

   layerMenu->Append(ID_ADD_LAYER,              _("Add Layer") + GetAccelerator(DO_ADD));
   layerMenu->Append(ID_CLONE,                  _("Clone Layer") + GetAccelerator(DO_CLONE));
   layerMenu->Append(ID_DUPLICATE,              _("Duplicate Layer") + GetAccelerator(DO_DUPLICATE));
   layerMenu->AppendSeparator();
   layerMenu->Append(ID_DEL_LAYER,              _("Delete Layer") + GetAccelerator(DO_DELETE));
   layerMenu->Append(ID_DEL_OTHERS,             _("Delete Other Layers") + GetAccelerator(DO_DELOTHERS));
   layerMenu->AppendSeparator();
   layerMenu->Append(ID_MOVE_LAYER,             _("Move Layer...") + GetAccelerator(DO_MOVELAYER));
   layerMenu->Append(ID_NAME_LAYER,             _("Name Layer...") + GetAccelerator(DO_NAMELAYER));
   layerMenu->Append(ID_SET_COLORS,             _("Set Layer Colors...") + GetAccelerator(DO_SETCOLORS));
   layerMenu->AppendSeparator();
   layerMenu->AppendCheckItem(ID_SYNC_VIEW,     _("Synchronize Views") + GetAccelerator(DO_SYNCVIEWS));
   layerMenu->AppendCheckItem(ID_SYNC_CURS,     _("Synchronize Cursors") + GetAccelerator(DO_SYNCCURS));
   layerMenu->AppendSeparator();
   layerMenu->AppendCheckItem(ID_STACK,         _("Stack Layers") + GetAccelerator(DO_STACK));
   layerMenu->AppendCheckItem(ID_TILE,          _("Tile Layers") + GetAccelerator(DO_TILE));
   layerMenu->AppendSeparator();
   layerMenu->AppendCheckItem(ID_LAYER0,        _("0"));
   // UpdateLayerItem will soon change the above item name

   helpMenu->Append(ID_HELP_INDEX,              _("Contents"));
   helpMenu->Append(ID_HELP_INTRO,              _("Introduction"));
   helpMenu->Append(ID_HELP_TIPS,               _("Hints and Tips"));
   helpMenu->Append(ID_HELP_ALGOS,              _("Algorithms"));
   helpMenu->Append(ID_HELP_LEXICON,            _("Life Lexicon"));
   helpMenu->Append(ID_HELP_ARCHIVES,           _("Online Archives"));
   helpMenu->AppendSeparator();
   helpMenu->Append(ID_HELP_PERL,               _("Perl Scripting"));
   helpMenu->Append(ID_HELP_PYTHON,             _("Python Scripting"));
   helpMenu->AppendSeparator();
   helpMenu->Append(ID_HELP_KEYBOARD,           _("Keyboard Shortcuts"));
   helpMenu->Append(ID_HELP_MOUSE,              _("Mouse Shortcuts"));
   helpMenu->AppendSeparator();
   helpMenu->Append(ID_HELP_FILE,               _("File Menu"));
   helpMenu->Append(ID_HELP_EDIT,               _("Edit Menu"));
   helpMenu->Append(ID_HELP_CONTROL,            _("Control Menu"));
   helpMenu->Append(ID_HELP_VIEW,               _("View Menu"));
   helpMenu->Append(ID_HELP_LAYER,              _("Layer Menu"));
   helpMenu->Append(ID_HELP_HELP,               _("Help Menu"));
   helpMenu->AppendSeparator();
   helpMenu->Append(ID_HELP_REFS,               _("References"));
   helpMenu->Append(ID_HELP_FORMATS,            _("File Formats"));
   helpMenu->Append(ID_HELP_BOUNDED,            _("Bounded Grids"));
   helpMenu->Append(ID_HELP_PROBLEMS,           _("Known Problems"));
   helpMenu->Append(ID_HELP_CHANGES,            _("Changes"));
   helpMenu->Append(ID_HELP_CREDITS,            _("Credits"));
   #ifndef __WXMAC__
      helpMenu->AppendSeparator();
   #endif
   // on the Mac the wxID_ABOUT item gets moved to the app menu
   helpMenu->Append(wxID_ABOUT,                 _("About Golly") + GetAccelerator(DO_ABOUT));
   
   // create the menu bar and append menus;
   // avoid using "&" in menu names because it prevents using keyboard shortcuts
   // like Alt+L on Linux
   wxMenuBar* menuBar = new wxMenuBar();
   if (menuBar == NULL) Fatal(_("Failed to create menu bar!"));
   menuBar->Append(fileMenu,     _("File"));
   menuBar->Append(editMenu,     _("Edit"));
   menuBar->Append(controlMenu,  _("Control"));
   menuBar->Append(viewMenu,     _("View"));
   menuBar->Append(layerMenu,    _("Layer"));
   #ifdef __WXMAC__
      // wxMac bug: need the "&" otherwise we get an extra Help menu
      menuBar->Append(helpMenu,  _("&Help"));
   #else
      menuBar->Append(helpMenu,  _("Help"));
   #endif
   
   #ifdef __WXMAC__
      // prevent Window menu being added automatically by wxMac 2.6.1+
      menuBar->SetAutoWindowMenu(false);
   #endif

   // attach menu bar to the frame
   SetMenuBar(menuBar);
}

// -----------------------------------------------------------------------------

void MainFrame::UpdateMenuAccelerators()
{
   // keyboard shortcuts have changed, so update all menu item accelerators
   wxMenuBar* mbar = GetMenuBar();
   if (mbar) {
      // wxMac bug: these app menu items aren't updated (but user isn't likely
      // to change them so don't bother trying to fix the bug)
      SetAccelerator(mbar, wxID_ABOUT,         DO_ABOUT);
      SetAccelerator(mbar, wxID_PREFERENCES,   DO_PREFS);
      SetAccelerator(mbar, wxID_EXIT,          DO_QUIT);
      
      SetAccelerator(mbar, ID_DRAW,            DO_CURSDRAW);
      SetAccelerator(mbar, ID_PICK,            DO_CURSPICK);
      SetAccelerator(mbar, ID_SELECT,          DO_CURSSEL);
      SetAccelerator(mbar, ID_MOVE,            DO_CURSMOVE);
      SetAccelerator(mbar, ID_ZOOMIN,          DO_CURSIN);
      SetAccelerator(mbar, ID_ZOOMOUT,         DO_CURSOUT);
      
      SetAccelerator(mbar, ID_SCALE_1,         DO_SCALE1);
      SetAccelerator(mbar, ID_SCALE_2,         DO_SCALE2);
      SetAccelerator(mbar, ID_SCALE_4,         DO_SCALE4);
      SetAccelerator(mbar, ID_SCALE_8,         DO_SCALE8);
      SetAccelerator(mbar, ID_SCALE_16,        DO_SCALE16);
      
      SetAccelerator(mbar, wxID_NEW,           DO_NEWPATT);
      SetAccelerator(mbar, wxID_OPEN,          DO_OPENPATT);
      SetAccelerator(mbar, ID_OPEN_CLIP,       DO_OPENCLIP);
      SetAccelerator(mbar, ID_SHOW_PATTERNS,   DO_PATTERNS);
      SetAccelerator(mbar, ID_PATTERN_DIR,     DO_PATTDIR);
      SetAccelerator(mbar, wxID_SAVE,          DO_SAVE);
      SetAccelerator(mbar, ID_SAVE_XRLE,       DO_SAVEXRLE);
      SetAccelerator(mbar, ID_RUN_SCRIPT,      DO_RUNSCRIPT);
      SetAccelerator(mbar, ID_RUN_CLIP,        DO_RUNCLIP);
      SetAccelerator(mbar, ID_SHOW_SCRIPTS,    DO_SCRIPTS);
      SetAccelerator(mbar, ID_SCRIPT_DIR,      DO_SCRIPTDIR);
      
      SetAccelerator(mbar, ID_UNDO,            DO_UNDO);
      SetAccelerator(mbar, ID_REDO,            DO_REDO);
      SetAccelerator(mbar, ID_NO_UNDO,         DO_DISABLE);
      SetAccelerator(mbar, ID_CUT,             DO_CUT);
      SetAccelerator(mbar, ID_COPY,            DO_COPY);
      SetAccelerator(mbar, ID_CLEAR,           DO_CLEAR);
      SetAccelerator(mbar, ID_OUTSIDE,         DO_CLEAROUT);
      SetAccelerator(mbar, ID_PASTE,           DO_PASTE);
      SetAccelerator(mbar, ID_PASTE_SEL,       DO_PASTESEL);
      SetAccelerator(mbar, ID_SELECTALL,       DO_SELALL);
      SetAccelerator(mbar, ID_REMOVE,          DO_REMOVESEL);
      SetAccelerator(mbar, ID_SHRINK,          DO_SHRINK);
      SetAccelerator(mbar, ID_RANDOM,          DO_RANDFILL);
      SetAccelerator(mbar, ID_FLIPTB,          DO_FLIPTB);
      SetAccelerator(mbar, ID_FLIPLR,          DO_FLIPLR);
      SetAccelerator(mbar, ID_ROTATEC,         DO_ROTATECW);
      SetAccelerator(mbar, ID_ROTATEA,         DO_ROTATEACW);
      
      SetAccelerator(mbar, ID_START,           DO_STARTSTOP);
      SetAccelerator(mbar, ID_NEXT,            DO_NEXTGEN);
      SetAccelerator(mbar, ID_STEP,            DO_NEXTSTEP);
      SetAccelerator(mbar, ID_RESET,           DO_RESET);
      SetAccelerator(mbar, ID_SETGEN,          DO_SETGEN);
      SetAccelerator(mbar, ID_FASTER,          DO_FASTER);
      SetAccelerator(mbar, ID_SLOWER,          DO_SLOWER);
      SetAccelerator(mbar, ID_SETBASE,         DO_SETBASE);
      SetAccelerator(mbar, ID_AUTO,            DO_AUTOFIT);
      SetAccelerator(mbar, ID_HYPER,           DO_HYPER);
      SetAccelerator(mbar, ID_HINFO,           DO_HASHINFO);
      SetAccelerator(mbar, ID_RECORD,          DO_RECORD);
      SetAccelerator(mbar, ID_DELTIME,         DO_DELTIME);
      SetAccelerator(mbar, ID_SETRULE,         DO_SETRULE);
      
      SetAccelerator(mbar, ID_FULL,            DO_FULLSCREEN);
      SetAccelerator(mbar, ID_FIT,             DO_FIT);
      SetAccelerator(mbar, ID_FIT_SEL,         DO_FITSEL);
      SetAccelerator(mbar, ID_MIDDLE,          DO_MIDDLE);
      SetAccelerator(mbar, ID_RESTORE00,       DO_RESTORE00);
      SetAccelerator(mbar, wxID_ZOOM_IN,       DO_ZOOMIN);
      SetAccelerator(mbar, wxID_ZOOM_OUT,      DO_ZOOMOUT);
      SetAccelerator(mbar, ID_TOOL_BAR,        DO_SHOWTOOL);
      SetAccelerator(mbar, ID_LAYER_BAR,       DO_SHOWLAYER);
      SetAccelerator(mbar, ID_EDIT_BAR,        DO_SHOWEDIT);
      SetAccelerator(mbar, ID_ALL_STATES,      DO_SHOWSTATES);
      SetAccelerator(mbar, ID_STATUS_BAR,      DO_SHOWSTATUS);
      SetAccelerator(mbar, ID_EXACT,           DO_SHOWEXACT);
      SetAccelerator(mbar, ID_GRID,            DO_SHOWGRID);
      SetAccelerator(mbar, ID_ICONS,           DO_SHOWICONS);
      SetAccelerator(mbar, ID_INVERT,          DO_INVERT);
      SetAccelerator(mbar, ID_BUFF,            DO_BUFFERED);
      SetAccelerator(mbar, ID_TIMELINE,        DO_SHOWTIME);
      SetAccelerator(mbar, ID_INFO,            DO_INFO);
      
      SetAccelerator(mbar, ID_ADD_LAYER,       DO_ADD);
      SetAccelerator(mbar, ID_CLONE,           DO_CLONE);
      SetAccelerator(mbar, ID_DUPLICATE,       DO_DUPLICATE);
      SetAccelerator(mbar, ID_DEL_LAYER,       DO_DELETE);
      SetAccelerator(mbar, ID_DEL_OTHERS,      DO_DELOTHERS);
      SetAccelerator(mbar, ID_MOVE_LAYER,      DO_MOVELAYER);
      SetAccelerator(mbar, ID_NAME_LAYER,      DO_NAMELAYER);
      SetAccelerator(mbar, ID_SET_COLORS,      DO_SETCOLORS);
      SetAccelerator(mbar, ID_SYNC_VIEW,       DO_SYNCVIEWS);
      SetAccelerator(mbar, ID_SYNC_CURS,       DO_SYNCCURS);
      SetAccelerator(mbar, ID_STACK,           DO_STACK);
      SetAccelerator(mbar, ID_TILE,            DO_TILE);
   }
}

// -----------------------------------------------------------------------------

void MainFrame::CreateDirControls()
{
   patternctrl = new wxGenericDirCtrl(splitwin, wxID_ANY, wxEmptyString,
                                      wxDefaultPosition, wxDefaultSize,
                                      #ifdef __WXMSW__
                                         // speed up a bit
                                         wxDIRCTRL_DIR_ONLY | wxNO_BORDER,
                                      #else
                                         wxNO_BORDER,
                                      #endif
                                      wxEmptyString   // see all file types
                                     );
   if (patternctrl == NULL) Fatal(_("Failed to create pattern directory control!"));

   scriptctrl = new wxGenericDirCtrl(splitwin, wxID_ANY, wxEmptyString,
                                     wxDefaultPosition, wxDefaultSize,
                                     #ifdef __WXMSW__
                                        // speed up a bit
                                        wxDIRCTRL_DIR_ONLY | wxNO_BORDER,
                                     #else
                                        wxNO_BORDER,
                                     #endif
                                     wxEmptyString
                                    );
   if (scriptctrl == NULL) Fatal(_("Failed to create script directory control!"));

   // With wxGTK 2.9, setting the filter in the wxGenericDirCtrl constructor
   // causes a crash, but setting it afterwards seems to work fine.
   scriptctrl->SetFilter(_T("Perl/Python scripts|*.pl;*.py"));
   
   #ifdef __WXMSW__
      // now remove wxDIRCTRL_DIR_ONLY so we see files
      patternctrl->SetWindowStyle(wxNO_BORDER);
      scriptctrl->SetWindowStyle(wxNO_BORDER);
   #endif

   #if defined(__WXGTK__)
      // make sure background is white when using KDE's GTK theme
      #if wxCHECK_VERSION(2, 9, 0)
         patternctrl->GetTreeCtrl()->SetBackgroundStyle(wxBG_STYLE_ERASE);
         scriptctrl->GetTreeCtrl()->SetBackgroundStyle(wxBG_STYLE_ERASE);
      #else
         patternctrl->GetTreeCtrl()->SetBackgroundStyle(wxBG_STYLE_COLOUR);
         scriptctrl->GetTreeCtrl()->SetBackgroundStyle(wxBG_STYLE_COLOUR);
      #endif
      patternctrl->GetTreeCtrl()->SetBackgroundColour(*wxWHITE);
      scriptctrl->GetTreeCtrl()->SetBackgroundColour(*wxWHITE);
      // reduce indent a bit
      patternctrl->GetTreeCtrl()->SetIndent(8);
      scriptctrl->GetTreeCtrl()->SetIndent(8);
   #elif defined(__WXMAC__)
      // reduce indent a bit more
      patternctrl->GetTreeCtrl()->SetIndent(6);
      scriptctrl->GetTreeCtrl()->SetIndent(6);
   #else
      // reduce indent a lot
      patternctrl->GetTreeCtrl()->SetIndent(4);
      scriptctrl->GetTreeCtrl()->SetIndent(4);
   #endif
   
   #ifdef __WXMAC__
      // reduce font size (to get this to reduce line height we had to
      // make a few changes to wxMac/src/generic/treectlg.cpp)
      wxFont font = wxSystemSettings::GetFont(wxSYS_DEFAULT_GUI_FONT);
      font.SetPointSize(12);
      patternctrl->GetTreeCtrl()->SetFont(font);
      scriptctrl->GetTreeCtrl()->SetFont(font);
   #endif
   
   if ( wxFileName::DirExists(patterndir) ) {
      // only show patterndir and its contents
      SimplifyTree(patterndir, patternctrl->GetTreeCtrl(), patternctrl->GetRootId());
   }
   if ( wxFileName::DirExists(scriptdir) ) {
      // only show scriptdir and its contents
      SimplifyTree(scriptdir, scriptctrl->GetTreeCtrl(), scriptctrl->GetRootId());
   }

   // install event handler to detect control/right-click on a file
   patternctrl->GetTreeCtrl()->Connect(wxID_ANY, wxEVT_LEFT_DOWN,
                                       wxMouseEventHandler(MainFrame::OnTreeClick));
   patternctrl->GetTreeCtrl()->Connect(wxID_ANY, wxEVT_RIGHT_DOWN,
                                       wxMouseEventHandler(MainFrame::OnTreeClick));
   scriptctrl->GetTreeCtrl()->Connect(wxID_ANY, wxEVT_LEFT_DOWN,
                                      wxMouseEventHandler(MainFrame::OnTreeClick));
   scriptctrl->GetTreeCtrl()->Connect(wxID_ANY, wxEVT_RIGHT_DOWN,
                                      wxMouseEventHandler(MainFrame::OnTreeClick));
   #ifdef __WXMSW__
      // fix double-click problem
      patternctrl->GetTreeCtrl()->Connect(wxID_ANY, wxEVT_LEFT_DCLICK,
                                          wxMouseEventHandler(MainFrame::OnTreeClick));
      scriptctrl->GetTreeCtrl()->Connect(wxID_ANY, wxEVT_LEFT_DCLICK,
                                         wxMouseEventHandler(MainFrame::OnTreeClick));
   #endif
}

// -----------------------------------------------------------------------------

// create the main window
MainFrame::MainFrame()
   : wxFrame(NULL, wxID_ANY, wxEmptyString, wxPoint(mainx,mainy), wxSize(mainwd,mainht))
{
   wxGetApp().SetFrameIcon(this);

   // initialize paths to some temporary files (in datadir so no need to be hidden);
   // they must be absolute paths in case they are used from a script command when the
   // current directory has been changed to the location of the script file
   clipfile = datadir + wxT("golly_clipboard");
   perlfile = datadir + wxT("golly_clip.pl");
   pythonfile = datadir + wxT("golly_clip.py");

   // create one-shot timer (see OnOneTimer)
   onetimer = new wxTimer(this, wxID_ANY);

   CreateMenus();
   CreateToolbar();
   
   // if tool bar is visible then adjust position of other child windows
   int toolwd = showtool ? toolbarwd : 0;

   int wd, ht;
   GetClientSize(&wd, &ht);
   // wd or ht might be < 1 on Windows
   if (wd < 1) wd = 1;
   if (ht < 1) ht = 1;

   // wxStatusBar can only appear at bottom of frame so we use our own
   // status bar class which creates a child window at top of frame
   // but to the right of the tool bar
   int statht = showexact ? STATUS_EXHT : STATUS_HT;
   if (!showstatus) statht = 0;
   statusptr = new StatusBar(this, toolwd, 0, wd - toolwd, statht);
   if (statusptr == NULL) Fatal(_("Failed to create status bar!"));
   
   // create a split window with pattern/script directory in left pane
   // and layer/edit/timeline bars and pattern viewport in right pane
   splitwin = new wxSplitterWindow(this, wxID_ANY,
                                   wxPoint(toolwd, statht),
                                   wxSize(wd - toolwd, ht - statht),
                                   #ifdef __WXMSW__
                                      wxSP_BORDER |
                                   #endif
                                   wxSP_3DSASH | wxSP_NO_XP_THEME | wxSP_LIVE_UPDATE);
   if (splitwin == NULL) Fatal(_("Failed to create split window!"));

   // create patternctrl and scriptctrl in left pane
   CreateDirControls();
   
   // create a window for right pane which contains layer/edit/timeline bars
   // and pattern viewport
   rightpane = new RightWindow(splitwin, 0, 0, wd - toolwd, ht - statht);
   if (rightpane == NULL) Fatal(_("Failed to create right pane!"));
   
   // create layer bar and initial layer
   CreateLayerBar(rightpane);
   AddLayer();
   
   // create edit bar
   CreateEditBar(rightpane);
   
   // create timeline bar
   CreateTimelineBar(rightpane);

   // enable/disable tool tips after creating bars with buttons
   #if wxUSE_TOOLTIPS
      wxToolTip::Enable(showtips);
      wxToolTip::SetDelay(1500);    // 1.5 secs
   #endif
   
   CreateTranslucentControls();     // must be done BEFORE creating viewport
   
   // create viewport at minimum size to avoid scroll bars being clipped on Mac
   int y = 0;
   if (showlayer) y += LayerBarHeight();
   if (showedit) y += EditBarHeight();
   viewptr = new PatternView(rightpane, 0, y, 40, 40,
                             wxNO_BORDER |
                             wxWANTS_CHARS |              // receive all keyboard events
                             wxFULL_REPAINT_ON_RESIZE |
                             wxVSCROLL | wxHSCROLL);
   if (viewptr == NULL) Fatal(_("Failed to create viewport window!"));
   
   // this is the main viewport window (tile windows have a tileindex >= 0)
   viewptr->tileindex = -1;
   bigview = viewptr;

   #if wxUSE_DRAG_AND_DROP
      // let users drop files onto viewport
      viewptr->SetDropTarget(new DnDFile());
   #endif
   
   // these seemingly redundant steps are needed to avoid problems on Windows
   splitwin->SplitVertically(patternctrl, rightpane, dirwinwd);
   splitwin->SetSashPosition(dirwinwd);
   splitwin->SetMinimumPaneSize(50);
   splitwin->Unsplit(patternctrl);
   splitwin->UpdateSize();

   splitwin->SplitVertically(scriptctrl, rightpane, dirwinwd);
   splitwin->SetSashPosition(dirwinwd);
   splitwin->SetMinimumPaneSize(50);
   splitwin->Unsplit(scriptctrl);
   splitwin->UpdateSize();

   if (showpatterns) splitwin->SplitVertically(patternctrl, rightpane, dirwinwd);
   if (showscripts) splitwin->SplitVertically(scriptctrl, rightpane, dirwinwd);

   InitDrawingData();         // do this after viewptr has been set

   pendingfiles.Clear();      // no pending script/pattern files
   command_pending = false;   // no pending command
   draw_pending = false;      // no pending draw
   keepmessage = false;       // clear status message
   generating = false;        // not generating pattern
   fullscreen = false;        // not in full screen mode
   showbanner = true;         // avoid first file clearing banner message
}

// -----------------------------------------------------------------------------

MainFrame::~MainFrame()
{
   delete onetimer;
   DestroyDrawingData();
}
