#include "WriteTrainingCsvFiles.h"
#include <boost/filesystem.hpp>
#include "TMath.h"

WriteTrainingCsvFiles::WriteTrainingCsvFiles():Tool(){}


bool WriteTrainingCsvFiles::Initialise(std::string configfile, DataModel &data){
  
  /////////////////// Useful header ///////////////////////
  if(configfile!="")  m_variables.Initialise(configfile); //loading config file
  //m_variables.Print();
  
  m_data= &data; //assigning transient data pointer
  
  // get configuration variables for this tool
  m_variables.Get("verbosity",verbosity);
  
  // Config Variables for this tool
  // ==============================
  std::string TrackLengthTrainingDataFile;
  std::string TrackLengthTestingDataFile;
  // first check if we're splitting the tchain loops into both training and testing
  get_ok = m_variables.Get("DNNTrainingEntries",trainingentries);
  if(not get_ok){
    auto errorlevel = (TrackLengthTestingDataFile!="NA") ? v_warning : v_message;
    Log("WriteTrainingCsvFiles Tool: No DNNTrainingEntries specified: no test file will be written",errorlevel,verbosity);
    trainingentries=-1;
  }
  // retrieve from m_data
  get_ok = m_variables.Get("TrackLengthTrainingDataFile",TrackLengthTrainingDataFile);
  if(not get_ok){
    Log("WriteTrainingCsvFiles Tool: No TrackLengthTrainingDataFile specified, will not be written",v_error,verbosity);
    return false;
  }
  get_ok = m_variables.Get("TrackLengthTestingDataFile",TrackLengthTestingDataFile);
  if(not get_ok){
    Log("WriteTrainingCsvFiles Tool: No TrackLengthTestingDataFile specified, will not be written",v_error,verbosity);
    return false;
  }
  get_ok = m_data->Stores.at("EnergyReco")->Get("MaxTotalHitsToDNN",maxhits0);
  if(not get_ok){
    Log("WriteTrainingCsvFiles Tool: No MaxTotalHitsToDNN in EnergyReco store!",v_error,verbosity);
    return false;
  }
  
  // Write the file header(s)
  // ========================
  if(trainingentries>1){
    // if we're splitting the run up into training and testing samples, we need to generate multiple output csvs
    tracklengthtrainingfiles.push_back(TrackLengthTrainingDataFile);
    tracklengthtrainingfiles.push_back(TrackLengthTestingDataFile);
  } else {
    // otherwise just one csv
    tracklengthtrainingfiles.push_back(TrackLengthTrainingDataFile);
  }
  
  // loop over the csv's we're creating and write header row
  for(std::string apath : tracklengthtrainingfiles){
    csvfile.open(apath,std::fstream::out);
    if(!csvfile.is_open()){
     Log("WriteTrainingCsvFiles Tool: Failed to open "+apath+" for writing headers",v_error,verbosity);
    }
    for (int i=0; i<maxhits0;++i){
       csvfile<<"l_"<<i<<",";
    }
    for (int i=0; i<maxhits0;++i){
       csvfile<<"T_"<<i<<",";
    }
    csvfile<<"lambda_max,"  //first estimation of track length(using photons projection on track)
           <<"totalPMTs,"   // number of PMT hits, not number of pmts.
           <<"totalLAPPDs," // number of LAPPD hits... 
           <<"lambda_max,"  // ... again...?
           <<"TrueTrackLengthInWater,"
           <<"neutrinoE,"
           <<"trueKE,"      // of the primary muon
           <<"diffDirAbs,"
           <<"TrueTrackLengthInMrd,"
           <<"recoDWallR,"
           <<"recoDWallZ,"
           <<"dirX,"        // of the reconstructed muon
           <<"dirY,"
           <<"dirZ,"
           <<"vtxX,"        // of the reconstructed event
           <<"vtxY,"
           <<"vtxZ,"
           <<"recoVtxFOM,"
           <<"recoStatus,"
           <<"deltaVtxR,"
           <<"deltaAngle"
           <<'\n';
    // }
    csvfile.close();
  }
  csvfile.open(tracklengthtrainingfiles.front(),std::fstream::app);  // open (first) file for writing events
  
  return true;
}

bool WriteTrainingCsvFiles::Execute(){
  Log("WriteTrainingCsvFiles Tool: Executing",v_message,verbosity);
  
  // Check if the event needs writing to file
  // ========================================
  uint32_t EventNumber;
  get_ok = m_data->Stores.at("ANNIEEvent")->Get("EventNumber", EventNumber);
  if(not get_ok){
    Log("WriteTrainingCsvFiles Tool: Failed to get EventNumber from ANNIEEvent",v_error,verbosity);
    return false;
  }
  // Not every event is written to file: check if this one passed checks:
  uint32_t ThisEvtNum;
  get_ok = m_data->Stores.at("EnergyReco")->Get("ThisEvtNum",ThisEvtNum);
  if(not get_ok){
    Log("WriteTrainingCsvFiles Tool: Failed to get ThisEvtNum from EneryReco store",v_error,verbosity);
    return false;
  }
  Log("WriteTrainingCsvFiles Tool: ThisEvtNum="+to_string(ThisEvtNum)+", current event is "+to_string(EventNumber),v_debug,verbosity);
  if(ThisEvtNum!=EventNumber){
    Log("WriteTrainingCsvFiles Tool: ThisEvtNum!=EventNumber; skipping this write",v_debug,verbosity);
    return true;
   } // this event didn't pass checks; don't write this entry
  
  // Retrieve variables from BoostStore
  // ==================================
  // First we need to declare the variables to fill
  std::vector<double> lambda_vector;
  std::vector<double> digitT;
  double lambda_max;
  int totalPMTs =0; // number of PMT hits in the event
  int totalLAPPDs = 0; // number of LAPPD hits in the event
  float TrueTrackLengthInWater2;
  double TrueNeutrinoEnergy;
  double trueEnergy;
  float diffDirAbs2;
  float TrueTrackLengthInMrd2;
  float recoDWallR2;
  float recoDWallZ2;
  Direction dirVec;
  Position vtxVec;
  double recoVtxFOM;
  Int_t recoStatus;
  double deltaVtxR;
  double deltaAngle;
  
  // Then do the retrieval TODO should check all these retrievals succeed
  Log("WriteTrainingCsvFiles Tool: Getting variables from EnergyReco Store",v_debug,verbosity);
  m_data->Stores.at("EnergyReco")->Get("lambda_vec",lambda_vector);
  m_data->Stores.at("EnergyReco")->Get("digit_ts_vec",digitT);
  m_data->Stores.at("EnergyReco")->Get("lambda_max",lambda_max);
  m_data->Stores.at("EnergyReco")->Get("num_pmt_hits",totalPMTs);
  m_data->Stores.at("EnergyReco")->Get("num_lappd_hits",totalLAPPDs);
  m_data->Stores.at("EnergyReco")->Get("TrueTrackLengthInWater",TrueTrackLengthInWater2);
  m_data->Stores.at("EnergyReco")->Get("trueNeuE",TrueNeutrinoEnergy);
  m_data->Stores.at("EnergyReco")->Get("trueE",trueEnergy);
  m_data->Stores.at("EnergyReco")->Get("diffDirAbs2",diffDirAbs2);
  m_data->Stores.at("EnergyReco")->Get("TrueTrackLengthInMrd2",TrueTrackLengthInMrd2);
  // FIXME naming, if training are these actually trueDWallR2?
  m_data->Stores.at("EnergyReco")->Get("recoDWallR2",recoDWallR2);
  m_data->Stores.at("EnergyReco")->Get("recoDWallZ2",recoDWallZ2);
  m_data->Stores.at("EnergyReco")->Get("dirVec",dirVec);
  m_data->Stores.at("EnergyReco")->Get("vtxVec",vtxVec);
  m_data->Stores.at("EnergyReco")->Get("recoVtxFOM",recoVtxFOM);
  m_data->Stores.at("EnergyReco")->Get("recoStatus",recoStatus);
  m_data->Stores.at("EnergyReco")->Get("deltaVtxR",deltaVtxR);
  m_data->Stores.at("EnergyReco")->Get("deltaAngle",deltaAngle);
  
  // Write to .csv file
  // ==================
  // pick which file to write to
  if((tracklengthtrainingfiles.size()>1) && (entries_written==trainingentries)){    // once we've processed requested
    Log("WriteTrainingCsvFile Tool: switching to testing file",v_debug,verbosity);
    csvfile.close();                                                           // number of training entries
    csvfile.open(tracklengthtrainingfiles.back(), std::fstream::app);          // switch output to testing file
    if(not csvfile.is_open()){
      Log("WriteTrainingCsvFile Tool: Failed to open testing file",v_debug,verbosity);
      return false;
    }
  }
  if(not csvfile.is_open()){
     Log("WriteTrainingCsvFile Tool: output file is closed, skipping write",v_debug,verbosity);
     return true;
  }
  for(int i=0; i<maxhits0;++i){
     csvfile<<lambda_vector.at(i)<<",";
  }
  for(int i=0; i<maxhits0;++i){
     csvfile<<digitT.at(i)<<",";
  }
  csvfile<<lambda_max<<","
         <<totalPMTs<<","
         <<totalLAPPDs<<","
         <<lambda_max<<","
         <<TrueTrackLengthInWater2<<","
         <<TrueNeutrinoEnergy<<","
         <<trueEnergy<<","
         <<diffDirAbs2<<","
         <<TrueTrackLengthInMrd2<<","
         <<recoDWallR2<<","
         <<recoDWallZ2<<","
         <<dirVec.X()<<","
         <<dirVec.Y()<<","
         <<dirVec.Z()<<","
         <<vtxVec.X()<<","
         <<vtxVec.Y()<<","
         <<vtxVec.Z()<<","
         <<recoVtxFOM<<","
         <<recoStatus<<","
         <<deltaVtxR<<","
         <<deltaAngle
         <<'\n';
  ++entries_written;
  
  return true;
}

bool WriteTrainingCsvFiles::Finalise(){
  if(csvfile.is_open()) csvfile.close();
  return true;
}
