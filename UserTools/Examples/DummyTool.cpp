#include "DummyTool.h"

DummyTool::DummyTool():Tool(){}


bool DummyTool::Initialise(std::string configfile, DataModel &data){

  if(configfile!="")  m_variables.Initialise(configfile);
  //m_variables.Print();

  m_data= &data;
 
  m_variables.Get("verbose",m_verbose);
  int state=99;
  m_data->CStore.Set("state",state);
 
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
