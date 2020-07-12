/* vim:set noexpandtab tabstop=4 wrap */
#ifndef GuiDemo_H
#define GuiDemo_H

#include <string>
#include <iostream>
#include <sstream>
#include <stdio.h>

#include "Tool.h"

// TUI
#include "imtui/imtui.h"
#include "imtui/imtui-impl-ncurses.h"
#include "ImGui_Logger.h"

// GUI
#include "imgui.h"
#include "imgui_impl_sdl.h"
#include "imgui_impl_opengl2.h"
#include <SDL.h>
#include <SDL_opengl.h>

/**
* \class GuiDemo
*
* This tool demonstrates the use of the ImGui framework to build a GUI or TUI interface within a Tool.
*
* $Author: M.O'Flaherty $
* $Date: 2020/07/17 $
* Contact: marcus.o-flaherty@warwick.ac.uk
*/

class GuiDemo: public Tool {
	
	public:
	
	GuiDemo(); ///< Simple constructor
	bool Initialise(std::string configfile,DataModel &data); ///< Initialise Function for setting up Tool resources. @param configfile The path and name of the dynamic configuration file to read in. @param data A reference to the transient data class used to pass information between Tools.
	bool Execute(); ///< Execute function used to perform Tool purpose.
	bool Finalise(); ///< Finalise function used to clean up resources.
	
	private:
	
	// =====================
	// Functions
	// =====================
	
	// User Functions
	// --------------
	bool ReadBoostStore();
	
	// GUI handling functions
	// ----------------------
	bool ImGui_Init();
	void ImGui_InitState();
	bool ImGui_BuildUI();
	void ImGui_Run();
	void ImGui_Render();
	bool ImGui_HandleSDLEvents();
	void ImGui_Suspend();
	void ImGui_Restore();
	void ImGui_Cleanup();
	
	// GUI window construction functions
	// ---------------------------------
	void BuildLogWindow();
	
	// Miscellaneous Functions
	// -----------------------
	// Log wrapper for switching to an in-TUI logging window when using a TUI
	// TODO: move this somewhere better...
	template <typename T> void ImGui_Log(T message, int messagelevel=1, int verbose=1){
		if(!tui){
			Log(message,messagelevel,verbose);
		} else {
			std::stringstream tmp;
			tmp<<message;
			tui_log.AddLog("[%d] %s\n", messagelevel,tmp.str().c_str());
		}
	}
	
	// =====================
	// Variables
	// =====================
	
	// GUI handling variables
	// ----------------------
	bool tui=false; // whether to use tui or gui
	// GUI variables...
	SDL_Window* window=nullptr;
	SDL_GLContext gl_context;
	ImGuiIO* io;
	// TUI variables...
	ImTui::TScreen* screen=nullptr;
	ExampleAppLog tui_log {"GuiDemo Log"};
	bool show_log_window;
	bool show_demo_window;
	
	// Gui state variables
	// -------------------
	bool need_reinit;
	ImVec4 clear_color;
	float fsliderval;
	int counter;
	int nframes;
	bool done_looking;
	bool quit;
	bool show_window;
	
	// Other variables
	// ---------------
	// verbosity levels: if 'verbosity' < this level, the message type will be logged.
	int verbosity=5;
	int v_error=0;
	int v_warning=1;
	int v_message=2;
	int v_debug=3;
	std::string logmessage;
	int get_ok;
};


#endif
