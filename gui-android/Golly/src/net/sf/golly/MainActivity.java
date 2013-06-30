/*** /

 This file is part of Golly, a Game of Life Simulator.
 Copyright (C) 2013 Andrew Trevorrow and Tomas Rokicki.

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

package net.sf.golly;

import java.io.File;

import android.app.Activity;
import android.graphics.Color;
import android.os.Bundle;
import android.os.Handler;
import android.view.Menu;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

public class MainActivity extends Activity {
    
	// see jnicalls.cpp for these routines:
    private static native void nativeClassInit();	// must be static
    private native void nativeCreate();				// must NOT be static
    private native void nativeDestroy();
    private native void nativeSetGollyDir(String path);
    private native void nativeSetTempDir(String path);
    private native void nativeSetSuppliedDirs(String prefix);
    private native void nativeStartStop();
    private native boolean nativeIsGenerating();
    private native String nativeGetStatusLine(int line);
    private native void nativeNewPattern();
    private native void nativeFitPattern();
    private native void nativeStep();

    // local fields
    private static boolean firstcall = true;
	private Button ssbutton;						// Start/Stop button
    private TextView status1, status2, status3;		// status bar has 3 lines
    private PatternGLSurfaceView pattView;			// OpenGL ES is used to draw patterns
    private Handler genhandler;						// for generating patterns
    private Runnable generate;						// code started/stopped by genhandler
    
    static {
    	System.loadLibrary("stlport_shared");	// must agree with Application.mk
        System.loadLibrary("golly");			// loads libgolly.so
        nativeClassInit();						// caches Java method IDs
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main_layout);
        ssbutton = (Button) findViewById(R.id.startstop);
        status1 = (TextView) findViewById(R.id.status1);
        status2 = (TextView) findViewById(R.id.status2);
        status3 = (TextView) findViewById(R.id.status3);
        /* this reduces font width, but a bit ugly:
        status1.setTextScaleX(0.9f);
        status2.setTextScaleX(0.9f);
        status3.setTextScaleX(0.9f);
        */
        
        genhandler = new Handler();
        generate = new Runnable() {
        	@Override
        	public void run() {
        		nativeStep();
        		genhandler.postDelayed(this, 1000/60);		// max speed is 60 fps???
        	}
        };

        if (firstcall) {
        	firstcall = false;
            InitPaths();		// sets tempdir, etc
        }
        nativeCreate();			// must be called every time (to cache jobject)
        
        // following will call the PatternGLSurfaceView constructor
        pattView = (PatternGLSurfaceView) findViewById(R.id.patternview);
        
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // this adds items to the action bar if it is present
        getMenuInflater().inflate(R.menu.main, menu);
        return true;
    }

    @Override
    protected void onPause() {
        super.onPause();
        pattView.onPause();
        genhandler.removeCallbacks(generate);
    }

    @Override
    protected void onResume() {
        super.onResume();
        pattView.onResume();
        updateButtons();
        if (nativeIsGenerating()) {
    		genhandler.post(generate);
    	}
    }

    @Override
    protected void onDestroy() {
    	genhandler.removeCallbacks(generate);		// already done in OnPause???
        nativeDestroy();
        super.onDestroy();
    }

    public void InitPaths() {
    	// first create subdirs if they don't exist
    	File filesdir = getFilesDir();
    	File subdir;
    	subdir = new File(filesdir, "Rules");		// for user's .rule files
    	subdir.mkdir();
    	subdir = new File(filesdir, "Saved");		// for saved pattern files
    	subdir.mkdir();
    	subdir = new File(filesdir, "Downloads");	// for downloaded pattern files
    	subdir.mkdir();
    	
    	// now set gollydir (parent of above subdirs)
		String gollydir = filesdir.getAbsolutePath();
		nativeSetGollyDir(gollydir);
    	
    	// set tempdir
    	File dir = getCacheDir();
    	String tempdir = dir.getAbsolutePath();
    	nativeSetTempDir(tempdir);
		
		// create subdirs for files supplied with app here!!!???
    	// (need to extract from assets??? use OBB???)
		subdir = getDir("Patterns", MODE_PRIVATE);
		subdir = getDir("Rules", MODE_PRIVATE);
		subdir = getDir("Help", MODE_PRIVATE);
		// subdir = /data/data/net.sf.golly/app_Help
		String prefix = subdir.getAbsolutePath();
		prefix = prefix.substring(0, prefix.length() - 4); // remove "Help"
		nativeSetSuppliedDirs(prefix);
    }

    public void DeleteTempFiles() {
    	File dir = getCacheDir();
    	File[] files = dir.listFiles();
    	if (files != null) {
    	    for (File file : files) file.delete();
    	}
    }

    public void updateButtons() {
    	if (nativeIsGenerating()) {
    		ssbutton.setText("Stop");
    		ssbutton.setTextColor(Color.rgb(255,0,0));
    	} else {
    		ssbutton.setText("Start");
    		ssbutton.setTextColor(Color.rgb(0,255,0));
    	}
    }
    
    // called when the Start/Stop button is tapped
    public void toggleStartStop(View view) {
    	nativeStartStop();
    	if (nativeIsGenerating()) {
    		// start generating immediately
    		genhandler.post(generate);
    	} else {
    		// stop generating
    		genhandler.removeCallbacks(generate);
    	}
    	updateButtons();
    }
    
    // called when the New button is tapped
    public void doNewPattern(View view) {
    	genhandler.removeCallbacks(generate);
    	nativeNewPattern();
    	updateButtons();
        // show current touch mode!!!???
        // updateDrawingState();
    	
    	// deletes all files in tempdir
    	//!!! only if numlayers == 1???
    	DeleteTempFiles();
    }
    
    // called when the Fit button is tapped
    public void doFitPattern(View view) {
    	nativeFitPattern();
    }

    // this method is called from C++ code (see jnicalls.cpp)
    private void RefreshPattern() {
    	pattView.requestRender();
    }

    // this method is called from C++ code (see jnicalls.cpp)
    private void ShowStatusLines() {
    	// this might be called from a non-UI thread
        runOnUiThread (new Runnable() {
            public void run() {
                status1.setText(nativeGetStatusLine(1));
                status2.setText(nativeGetStatusLine(2));
                status3.setText(nativeGetStatusLine(3));
        	}
        });
    }
}