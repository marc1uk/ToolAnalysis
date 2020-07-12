/* vim:set noexpandtab tabstop=4 wrap */
#include "GuiDemo.h"
#include <chrono>
#include <thread>

GuiDemo::GuiDemo():Tool(){}

// TODO put all boilerplate code into a pure virtual class,
// so that GUI tools can derive from that class and need only provide
// the definitions for ImGui_BuildUI and optionally ImGui_InitState

bool GuiDemo::Initialise(std::string configfile, DataModel &data){
	
	std::cout<<"GuiDemo Tool: Initializing"<<std::endl;
	/////////////////// Useful header ///////////////////////
	 // load config file
	if(configfile!="") m_variables.Initialise(configfile);
	//m_variables.Print();
	
	//assign transient data pointer
	m_data= &data;
	/////////////////////////////////////////////////////////////////
	
	m_variables.Get("verbosity",verbosity);
	m_variables.Get("tui",tui);
	
	Log("GuiDemo Tool: Initializing GUI",v_debug,verbosity);
	ImGui_Init();
	
	return true;
}

bool GuiDemo::Execute(){
	/*
	                       XXX NOTE!!! XXX                         
	   cout/cerr/clog and Log() calls within Execute() will be
	   swallowed and will corrupt the screen while TUI is in use!
	   Instead, within the Execute() body please use 'ImGui_Log(...)'
	   which will either invoke the normal Log() function if using GUI,
	   or will append to an in-TUI log window when using the TUI.
	*/
	ImGui_Log("GuiDemo Tool: Executing",v_message,verbosity);
	
	// retrieve the data about the next event from the BoostStore
	ImGui_Log("GuiDemo Tool: Reading BoostStore",v_debug,verbosity);
	ReadBoostStore();
	
	// Interactive Loop - returns when the user sets `done_looking` or `quit` to true
	// Use `done looking` to continue the ToolChain execution
	// Use `quit` to terminate it. With a GUI this is automatically called if you close the GUI.
	// These booleans must be attached to buttons built into your GUI!
	ImGui_Run();
	
	if(quit){
		// if user closed the window, we need to end the toolchain (... do we?)
		ImGui_Log("GuiDemo Tool: User quit, ending ToolChain",v_debug,verbosity);
		m_data->vars.Set("StopLoop",1);
	}
	
	ImGui_Log("GuiDemo Tool: Finished",v_debug,verbosity);
	return true;
}

bool GuiDemo::Finalise(){
	ImGui_Cleanup();
	return true;
}

bool GuiDemo::ReadBoostStore(){
	// TODO
	return true;
}

// #######################
// ## Building GUI code ##
// #######################

// GUI State Variables
// ===================
void GuiDemo::ImGui_InitState(){
	// Initialize any UI state variables that will define how we build the UI.
	// Things like what windows are visible, what tree nodes are open, etc.
	// This only gets called when the UI is first opened - its state then persists
	// subsequent Execute() calls until the toolchain closes.
	ImGui_Log("GuiDemo Tool: Initializing GUI state variables",v_debug,verbosity);
	
	fsliderval = 0.0f;
	counter = 0;
	nframes = 0;
	show_window = true;         // show our main window
	show_log_window = false;    // show the ImGui_Log window
	show_demo_window = false;   // show the ImGui demo window
}

// Build/Update GUI
// ================
bool GuiDemo::ImGui_BuildUI(){
	// Somewhat strangely building and updating the UI are the same
	// - with each refresh we appear to build the GUI from scratch
	
	// Create a window called "Hello, world!"
	// We can hard-code the initial size and position of the window.
	// I think this is a fall-back if there is no imgui.ini...
	//ImGui::SetNextWindowPos(ImVec2(4, 2), ImGuiCond_Once);
	//ImGui::SetNextWindowSize(ImVec2(50.0, 10.0), ImGuiCond_Once);
	
	// the above should be called before 'ImGui::Begin()', which constructs a new window
	//ImGui::Begin("Closer");
	
	// then populate the window contents
	//ImGui::Checkbox("Show Main Window: ", &show_window);
	
	// when you're done, call ImGui::End
	//ImGui::End();
	
	// Here's a more fully fledged version
	if(show_window){
		ImGui::Begin("Hello, world!", &show_window);
		
		// a simple text line
		ImGui::Text("NFrames = %d", nframes++);
		
		// Some checkboxes to control the bools storing our window open/close state
		ImGui::Checkbox("Done Looking", &done_looking);
		ImGui::Checkbox("Show Tool Log", &show_log_window);
		ImGui::Checkbox("Show Demo Window", &show_demo_window);
		
		// a float slider from 0.0f to 1.0f
		// note some control parameters (primarily relating to size) may need tuning between GUI/TUI
		ImGui::Text("Float:");
		ImGui::SameLine();
		if(!tui){
			ImGui::SliderFloat("float", &fsliderval, 0.0f, 1.0f);
		} else {
			ImGui::SliderFloat("float", &fsliderval, 0.0f, 10.0f);
		}
		
		// Buttons return true when clicked
		if (ImGui::Button("Button")) counter++;
		ImGui::Text("counter = %d", counter);
		
		// random others
		ImGui::Text("Mouse Pos : x = %g, y = %g", ImGui::GetIO().MousePos.x, ImGui::GetIO().MousePos.y);
		ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate,
			 ImGui::GetIO().Framerate);
		
		ImGui::End();
	}
	// In practice it would be sensible for each window to be defined in its own function
	// each window is then shown (or not, if desired) by calling that function
	
	// Window to display Tool log messages
	if(show_log_window) BuildLogWindow();
	
	// ImGui demo window with ALL the widgets!
	if(show_demo_window) ImGui::ShowDemoWindow();
	
	return true;
}

// ######################################################################
// ## Boilerplate code - User need not touch anything below this point ##
// ######################################################################
// This is boilerplate code from imgui/examples/example_sdl_opengl2
// and imgui/examples/example_imtui, with assorted modifications
// TODO find a better place for this to live.
// ======================================================================

bool GuiDemo::ImGui_Init(){
	// boilerplate setup code
	if(!tui){
		// Setup SDL for GUI
		if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER | SDL_INIT_GAMECONTROLLER) != 0){
			std::cout<<"GuiDemo Tool: Error loading SDL: "+std::string(SDL_GetError())<<std::endl;;
			return false;
		}
		
		// Setup window
		SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
		SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
		SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);
		SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
		SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);
		SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);
		window = SDL_CreateWindow("Dear ImGui SDL2+OpenGL example", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 1280, 720, window_flags);
		gl_context = SDL_GL_CreateContext(window);
		SDL_GL_MakeCurrent(window, gl_context);
		SDL_GL_SetSwapInterval(1); // Enable vsync
	}
	
	// Setup Dear ImGui context
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	io = &ImGui::GetIO(); //(void)io;
	
	if(!tui){
		// Setup Dear ImGui style
		ImGui::StyleColorsDark();
		//ImGui::StyleColorsClassic();
		
		// Setup Platform/Renderer bindings
		ImGui_ImplSDL2_InitForOpenGL(window, gl_context);
		ImGui_ImplOpenGL2_Init();
	} else {
		// setup ncurses for TUI
		screen = ImTui_ImplNcurses_Init(true);
		ImTui_ImplText_Init();
	}
	
	// Initialize UI state (collapse trees, default views etc)
	need_reinit=false;
	quit = false;
	ImGui_InitState();  // custom, user should populate this
	
	return true;
}

void GuiDemo::ImGui_Run(){
	
	if(tui) ImGui_Log("GuiDemo Tool: Restoring TUI",v_debug,verbosity);
	ImGui_Restore(); // Restore TUI if suspended between Execute calls
	// FIXME If this tool errors out we MUST call ImGui_Suspend() or it'll ruin the console state!
	
	// Encapsulate the interaction loop, since not much can happen in here anyway
	ImGui_Log("GuiDemo Tool: Entering iterative UI loop",v_debug,verbosity);
	// N.B: This loop will execute many times! No print statements please.
	done_looking=false;
	while(!done_looking){
		// handle closing of the GUI window... and others?
		if(!tui) quit = ImGui_HandleSDLEvents();
		// if the user closed the gui window, terminate
		if(quit){ break; }
		
		// Start the Dear ImGui frame - boilerplate
		if(!tui){
			ImGui_ImplOpenGL2_NewFrame();
			ImGui_ImplSDL2_NewFrame(window);
		} else {
			ImTui_ImplNcurses_NewFrame();
			ImTui_ImplText_NewFrame();
		}
		ImGui::NewFrame();
		
		// update the GUI
		ImGui_BuildUI();
		
		// render updates to the screen
		ImGui_Render();
	}
	ImGui_Log("GuiDemo Tool: End of interactive loop",v_debug,verbosity);
	
	if(tui) ImGui_Log("GuiDemo Tool: Suspending TUI",v_debug,verbosity);
	ImGui_Suspend(); // suspend TUI between execute calls so other tool outputs work as normal
	return;
}

void GuiDemo::ImGui_Render(){
	// Boilerplate rendering code
	ImGui::Render();
	
	if(!tui){
		glViewport(0, 0, (int)io->DisplaySize.x, (int)io->DisplaySize.y);
		glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
		glClear(GL_COLOR_BUFFER_BIT);
		ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());
		SDL_GL_SwapWindow(window);
	} else {
		ImTui_ImplText_RenderDrawData(ImGui::GetDrawData(), screen);
		ImTui_ImplNcurses_DrawScreen();
	}
}

bool GuiDemo::ImGui_HandleSDLEvents(){
	// boilerplate code to get GUI events
	SDL_Event event;
	while (SDL_PollEvent(&event)){
		ImGui_ImplSDL2_ProcessEvent(&event);
		if (event.type == SDL_QUIT)
		quit = true;
	}
	return quit;
}

void GuiDemo::ImGui_Cleanup(){
	// boilerplate cleanup code
	
	if(!tui){
		//std::cout<<"Shutting down ImGui implementation"<<std::endl;
		ImGui_ImplOpenGL2_Shutdown();
		ImGui_ImplSDL2_Shutdown();
	} else {
		// one last rebuild and render, which helps to capture mouse events
		// immediately before cleanup. Without this, if the user moves the mouse
		// just as ncurses is being shutdown, it can end up spewing some
		// nonsense into the terminal.
		ImGui_Render();
		// now the actual shutdown code
		ImTui_ImplText_Shutdown();
		ImTui_ImplNcurses_Shutdown();
	}
	//std::cout<<"Destroying ImGui context"<<std::endl;
	ImGui::DestroyContext();
	
	if(!tui){
		//std::cout<<"Destroying SDL context"<<std::endl;
		if(gl_context) SDL_GL_DeleteContext(gl_context);
		if(window) SDL_DestroyWindow(window);
		SDL_Quit();
	}
	return;
}

// I added these functions for suspending and resuming the ncurses TUI as part of a ToolChain
void GuiDemo::ImGui_Suspend(){
	if(tui) ImTui_ImplNcurses_Suspend();
	need_reinit = true;
}

void GuiDemo::ImGui_Restore(){
	if(tui && need_reinit) ImTui_ImplNcurses_Restore();
	need_reinit = false;
}

// Using the TUI takes control of the console, so we need another way to make logging messages
// visible to the user. One option is to use a TUI window that the user can open to view logs.
void GuiDemo::BuildLogWindow(){
	// sane position and size of the window if the last run used GUI rather than TUI
	ImGui::SetNextWindowPos(ImVec2(20, 14), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(100, 30), ImGuiCond_FirstUseEver);
	
	// constructs the ImGui window and populates it
	tui_log.Draw(&show_log_window);
}
