#ifndef WriteTrainingCsvFiles_H
#define WriteTrainingCsvFiles_H

#include <string>
#include <iostream>
#include <sstream>
#include "Tool.h"


/**
 * \class WriteTrainingCsvFiles
 *
 * This is a blank template for a Tool used by the script to generate a new custom tool. Please fill out the descripton and author information.
*
* $Author: B.Richards $
* $Date: 2019/05/28 10:44:00 $
* Contact: b.richards@qmul.ac.uk
*/
class WriteTrainingCsvFiles: public Tool {
  
  public:
  WriteTrainingCsvFiles(); ///< Simple constructor
  bool Initialise(std::string configfile,DataModel &data); ///< Initialise Function for setting up Tool resorces. @param configfile The path and name of the dynamic configuration file to read in. @param data A reference to the transient data class used to pass information between Tools.
  bool Execute(); ///< Executre function used to perform Tool perpose. 
  bool Finalise(); ///< Finalise funciton used to clean up resorces.
  
  private:
  int maxhits0=1100;
  std::ofstream csvfile;
  std::vector<std::string> tracklengthtrainingfiles;
  long long trainingentries;
  uint32_t entries_written=0;
  
  // verbosity levels: if 'verbosity' < this level, the message type will be logged.
  int verbosity=1;
  int v_error=0;
  int v_warning=1;
  int v_message=2;
  int v_debug=3;
  std::string logmessage;
  int get_ok;

};


#endif
