#include "DummyTool.h"

DummyTool::DummyTool():Tool(){}


bool DummyTool::Initialise(std::string configfile, DataModel &data){

  if(configfile!="")  m_variables.Initialise(configfile);
  //m_variables.Print();

  m_data= &data;
 
  m_variables.Get("verbose",m_verbose);
  
  // shove some stuff in the CStore for testing the pythonAPI
  int state=99;
  m_data->CStore.Set("state",state);
  std::string astring="potatoes";
  m_data->CStore.Set("astring",astring);
  std::vector<double> dubs{1.,2.,3.};
  m_data->CStore.Set("dubs",dubs);
  Position pos(3.,2.,1.);
  m_data->CStore.Set("pos",pos);
  
  // shove some stuff in vars ASCII store for testing the pythonAPI
  m_data->vars.Set("state",state);
  m_data->vars.Set("astring",astring);
 
  Log("test 1",1,m_verbose);

  return true;
}


bool DummyTool::Execute(){
  
  Log("test 2",2,m_verbose);

  return true;
}


bool DummyTool::Finalise(){

  return true;
}
