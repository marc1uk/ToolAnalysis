/* vim:set noexpandtab tabstop=4 wrap filetype=cpp */
#include "Postgres.h"
#include "DataModel.h"
#include <signal.h>

void Postgres::SetVerbosity(int verb){
	verbosity=verb;
}

// TODO probably could do better with the exception handling throughout this file
pqxx::connection* Postgres::OpenConnection(){
	if(verbosity>v_debug) std::cout<<"Opening Connection"<<std::endl;
	try{
		// if we already have a connection open, nothing to do
		if(conn!=nullptr && conn->is_open()){
			if(verbosity>v_debug) std::cout<<"Connection already open"<<std::endl;
			return conn;
		}
		
		// otherwise form the connection string
		std::stringstream tmp;
		
		dbname="postgres";
		port=5434;
		//hostname="/var/run/postgres";
		hostname="/tmp";
		
		tmp<<"dbname="<<dbname<<" port="<<port<<" user="<<dbuser<<" password="<<password;
		if(hostname==""){
			tmp << " hostaddr="<<hostaddr;
		} else {
			tmp << " host="<<hostname;
		}
		
		// attempt to connect to the database
		if(verbosity>v_debug) std::cout<<"connecting with string '"<<tmp.str()<<"'"<<std::endl;
		conn = new pqxx::connection(tmp.str().c_str());
		
		// verify we succeeded
		// "don't use is_open(), use the broken_connection exception", they say. Hmm.
		// but will that be thrown now, or only when we try to *use* the connection, for a transaction?
		if(!conn->is_open()){
			std::cerr<<"Failed to connect to the database! Connection string was: '"
			         <<tmp.str()<<"', please verify connection details"<<std::endl;
			return nullptr;
		}
		return conn;
	}
	catch (const pqxx::broken_connection &e){
		// as usual the doxygen sucks, but it seems this doesn't provide
		// any further methods to obtain information about the failure mode,
		// so probably not useful to catch this explicitly.
		std::cerr << e.what() << std::endl;
	}
	catch (std::exception const &e){
		std::cerr << e.what() << std::endl;
	}
	return nullptr;
}

bool Postgres::CloseConnection(){
	if(verbosity>v_debug){
		std::cout<<"Closing connection"<<std::endl;
	}
	if(conn==nullptr){
		if(verbosity>v_debug) std::cout<<"No connection to close"<<std::endl;
		return true;
	}
	try {
		if(!conn->is_open()){
			if(verbosity>v_debug) std::cout<<"Connection is not open"<<std::endl;
			return true;
		}
		conn->disconnect();
		if(conn->is_open()){
			std::cerr<<"Attempted to close postgresql connection, yet it remains open?!" <<std::endl;
			return false;
		}
		return true;
	}
	catch (const pqxx::broken_connection &e){
		return true; // sure
	}
	catch (std::exception const &e){
		std::cerr << e.what() << std::endl;
		return false; // umm....
	}
	return false; // dummy
}

Postgres::~Postgres(){
	CloseConnection();
	if(conn) delete conn;
}

Postgres::Postgres(DataModel* m_data_in){
	m_data = m_data_in;
	// apparently when a connection breaks, it not only throws an exception
	// but sends a SIGPIPE signal that by default will kill the application!
	// https://libpqxx.readthedocs.io/en/6.3/a00915.html
	// let's please not do that.
	signal(SIGPIPE, SIG_IGN);
}

bool Postgres::GetCurrentRun(bool update, int* runnum_in, int* runconfig_in){
	if(verbosity>v_debug) std::cout<<"Getting current run"<<std::endl;
	// get the current (assumed latest) run number from the sql database
	try {
		// ensure we have an open connection to the database
		OpenConnection();
		
		// open a transaction to interact with the database
		//pqxx::work(*conn);
		if(verbosity>v_debug) std::cout<<"Opening nontransaction"<<std::endl;
		pqxx::nontransaction txn(*conn);
		
		// query the latest run number from the run database
		std::string query_string = "SELECT max(runnum) FROM run;";
		if(verbosity>v_debug) std::cout<<"Querying max run number"<<std::endl;
		pqxx::row row = txn.exec1(query_string);
		int runnum = row[0].as<int>();
		if(verbosity>v_debug) std::cout<<"Max run number is "<<runnum<<std::endl;
		
		// get its corresponding run type
		query_string = "SELECT runconfig FROM run WHERE runnum = " + pqxx::to_string(runnum);
		if(verbosity>v_debug) std::cout<<"Getting corresponding runconfig ID"<<std::endl;
		row = txn.exec1(query_string);  // TODO better check for 1 row?
		int runconfig = row[0].as<int>();
		if(verbosity>v_debug) std::cout<<"runconfig ID is "<<runconfig<<std::endl;
		
		// if 'update', update the values in the datamodel
		m_data->vars.Set("runnum",runnum);
		m_data->vars.Set("runconfig",runconfig);
		
		// if given variables to store the result in, set them
		if(runnum_in) *runnum_in = runnum;
		if(runconfig_in) *runconfig_in = runconfig;
		// return success
		return true;
	}
	catch (const pqxx::syntax_error &e){
		std::cerr << e.what() << std::endl
		          << "When executing query: " << e.query()
		          << " at position " << e.error_position;
		if(e.sqlstate()!=""){
			std::cerr << ", with SQLSTATE error code: " << e.sqlstate();
		}
		std::cerr << std::endl;
	}
	catch (const pqxx::sql_error &e){
		std::cerr << e.what() << std::endl
		          << "When executing query: " << e.query();
		if(e.sqlstate()!=""){
			std::cerr << ", with SQLSTATE error code: " << e.sqlstate();
		}
		std::cerr<<std::endl;
	}
	catch (std::exception const &e){
		std::cerr << e.what() << std::endl;
	}
	return false;
}

bool Postgres::GetSystemConfig(bool update, int* systemconfig_in, std::string systemname, int runconfig){
	if(verbosity>v_debug) std::cout<<"Getting System config ID"<<std::endl;
	// get the id for the system configuration for a given run
	// if no system given, assume we're asking for the current system, which should be stored in m_data
	if(systemname==""){
		if(verbosity>v_debug) std::cout<<"using default system name"<<std::endl;
		get_ok = m_data->vars.Get("system",systemname);
		if(not get_ok){
			std::cerr<<"Postgres::GetSystemConfig error! no system name given and none in m_data!"<<std::endl;
			return false;
		}
		if(verbosity>v_debug) std::cout<<"system: "<<systemname<<std::endl;
	}
	// if no runconfig is given, assume we're asking about the runconfig for the current run type.
	if(runconfig<0){
		if(verbosity>v_debug) std::cout<<"Getting run configuration ID"<<std::endl;
		// the current run type should be in m_data...
		get_ok = m_data->vars.Get("runconfig",runconfig);
		if(not get_ok){
			// but if not, we can try to get it via GetCurrentRun()
			if(verbosity>v_debug) std::cout<<"None given, getting configuration for latest run"<<std::endl;
			get_ok = GetCurrentRun();
			if(not get_ok){
				std::cerr<<"Postgres::GetSystemConfig error! no runconfig given, none in m_data,"
				         <<" and failed to retrieve one based on the current run number!"<<std::endl;
				return false;
			}
		}
		// we should have a value in the datamodel at this point
		get_ok = m_data->vars.Get("runconfig",runconfig);
		if(not get_ok){
			// ..??! this is the kind of place you'd put 'assert(get_ok==true)'
			std::cerr<<"Postgres::GetSystemConfig error! GetCurrentRun called but still no runconfig in m_data!"
			         <<std::endl;
			return false;
		}
		if(verbosity>v_debug) std::cout<<"run configuration ID "<<runconfig<<std::endl;
	}
	try {
		// ensure we have an open connection to the database
		OpenConnection();
		
		// open a transaction to interact with the database
		if(verbosity>v_debug) std::cout<<"Opening nontransaction"<<std::endl;
		//pqxx::work(*conn);
		pqxx::nontransaction txn(*conn);
		
		// based on the system name and run type (i.e. runconfig), look up the system configuration number
		// columns in the runconfig table follow their system name, e.g. 'daq'/'vme'/'hv'
		// XXX note the use of txn.quote_name instead of txn.quote when quoting table/column identifiers!!!
		std::string query_string = std::string("SELECT ") +txn.quote_name(systemname)
		                           +" FROM runconfig WHERE id = "+pqxx::to_string(runconfig);
		// run the query, get the system configuration number
		if(verbosity>v_debug) std::cout<<"Querying system configuration ID with query: \n"<<query_string<<"\n";
		pqxx::row row = txn.exec1(query_string);
		int systemconfig = row[0].as<int>();
		if(verbosity>v_debug) std::cout<<"system config ID: "<<systemconfig<<std::endl;
		
		// if 'update', update the values in the datamodel
		m_data->vars.Set("systemconfig",systemconfig);
		
		// if given a variable to store the result in, set it
		if(systemconfig_in) *systemconfig_in = systemconfig;
		
		// return success
		return true;
	}
	catch (const pqxx::syntax_error &e){
		std::cerr << e.what() << std::endl
		          << "When executing query: " << e.query()
		          << " at position " << e.error_position;
		if(e.sqlstate()!=""){
			std::cerr << ", with SQLSTATE error code: " << e.sqlstate();
		}
		std::cerr << std::endl;
	}
	catch (const pqxx::sql_error &e){
		std::cerr << e.what() << std::endl
		          << "When executing query: " << e.query();
		if(e.sqlstate()!=""){
			std::cerr << ", with SQLSTATE error code: " << e.sqlstate();
		}
		std::cerr<<std::endl;
	}
	catch (std::exception const &e){
		std::cerr << e.what() << std::endl;
	}
	
	return false;
}

std::string Postgres::GetToolConfig(std::string toolname, int versionnum, std::string systemname){
	
	try {
		// ensure we have an open connection to the database
		OpenConnection();
		
		// to uniquely identify a config file within the configfiles table
		// we need to provide the system name (e.g. "DAQ", "VME", "MRD"...)
		// the tool name, and the version number. Including the system name allows there to be
		// different tools with the same name under each different system's codebase.
		// if no systemname is given, we assume the user means the current system,
		// the name of which should be in the datamodel.
		if(systemname==""){
			get_ok = m_data->vars.Get("system",systemname);
			if(not get_ok){
				std::cerr<<"Postgres::GetToolConfig error! no system name given and none in m_data!"<<std::endl;
				return false;
			}
		}
		
		// if no version number is given, we assume the user means the version number
		// described by the runconfig for the current run, which should be in m_data
		if(versionnum<0){
			if(verbosity>v_debug){
				std::cout<<"No version number for requested Tool config, looking up from "
				         <<"latest run configuration"<<std::endl;
			}
			int runconfig;
			get_ok = m_data->vars.Get("runconfig", runconfig);
			if(not get_ok){
				// if there's no runconfig in the datamodel we can try to get one
				// based on the current run number - or if this isn't held locally,
				// based on the maximum run number in the database run table
				get_ok = GetSystemConfig();
				if(not get_ok){
					std::cerr<<"Postgres::GetToolConfig error! "
					         <<"no version given and failed calling GetSystemConfig!"<<std::endl;
					return "";
				}
				// else we should have one in the datamodel now
				get_ok = m_data->vars.Get("runconfig", runconfig);
				if(not get_ok){
					// ...?! this is where you'd call 'assert(get_ok)'
					std::cerr<<"Postgres::GetToolConfig called GetSystemConfig but no runconfig in datamodel!"
						     <<std::endl;
					return "";
				}
			}
			if(verbosity>v_debug) std::cout<<"run configuration "<<runconfig<<std::endl;
			
			// we should have a valid system name and runconfig now. with these we can
			// retrieve the list of tools and their corresponding config file version numbers
			// open a transaction to interact with the database
			if(verbosity>v_debug) std::cout<<"Opening nontransaction"<<std::endl;
			//pqxx::work(*conn);
			pqxx::nontransaction txn(*conn);
			
			// build the query
			std::string query_string = "SELECT configfiles FROM "+txn.quote_name(systemname)
			      +" WHERE id = "+pqxx::to_string(runconfig)+";";
			// perform the query
			if(verbosity>v_debug){
				std::cout<<"Getting system configuration ID with query \n"<<query_string<<"\n";
			}
			pqxx::result res = txn.exec(query_string);
			// check we had a result
			if(res.size()==0){
				std::cerr<<"Postgres::GetToolConfig found no record in "<<systemname
					     <<" with id "<<runconfig<<std::endl;
				return "";
			}
			// check we did not find more than one record (each runconfig's id should be unique)
			if(res.size()>1){
				std::cerr<<"Postgres::GetToolConfig error! found more than one record in "<<systemname
					     <<" with id "<<runconfig<<"! This should not happen! Verify database integrity!"
					     <<std::endl;
				return "";
			}
			// extract the list of tools and their configfile versions. This should be a json string.
			if(verbosity>v_debug) std::cout<<"extracting system configuration as json"<<std::endl;
			std::string json_string = res[0][0].as<string>();
			if(verbosity>v_debug) std::cout<<"result is : "<<json_string<<std::endl;
			// build a store from it for easy parsing
			if(verbosity>v_debug) std::cout<<"parsing system configuration"<<std::endl;
			Store configfiles;
			configfiles.JsonParser(json_string);
			// try to get the configfile version number for our specified tool
			if(not configfiles.Get(toolname,versionnum)){
				std::cerr<<"Postgres::GetToolConfig error! attempt to get configfile version number"
						 <<" for tool "<<toolname<<" in run config with id "<<runconfig
						 <<" did not find this tool! Are you sure the tool is supposed to be in the toolchain?"
						 <<std::endl;
				return "";
			}
			if(verbosity>v_debug){
				std::cout<<"Tool "<<toolname<<" configfile version was "<<versionnum<<std::endl;
			}
			// sanity check
			if(versionnum<0){
				std::cerr<<"Postgres::GetToolConfig error! invalid version number "<<versionnum
					     <<", looked up from runconfig id "<<runconfig<<std::endl;
				return "";
			}
		}
		
		// ok! we should now have all the ingredients we need to get the confile file contents
		// open a transaction to interact with the database
		// (the previous one will have been closed when it went out of scope, if it was opened)
		if(verbosity>v_debug) std::cout<<"Opening nontransaction"<<std::endl;
		//pqxx::work(*conn);
		pqxx::nontransaction txn(*conn);
		
		// build the query
		std::string query_string = "SELECT contents FROM configfiles WHERE name = "
		    +txn.quote(toolname)+" AND system = " + txn.quote(systemname)
		    + " AND version = " + pqxx::to_string(versionnum);
		
		// attempt to run the query
		if(verbosity>v_debug){
			std::cout<<"Getting configfile contents with query \n"<<query_string<<"\n";
		}
		pqxx::result res = txn.exec(query_string);
		
		// check we have a result
		if(res.size()==0){
			std::cerr<<"Could not find a matching configfiles entry for tool "
			         <<toolname<<" with version number "<<versionnum<<std::endl;
			return "";
		}
		
		// check we have only one result
		if(res.size()>1){
			std::cerr<<"Found more than one matching configfile table entry for tool "
			         <<toolname<<" with version number "<<versionnum
			         <<", this should not be possible! Verify database integrity!"<<std::endl;
			return "";
		}
		
		// else we have just one result, and it should have just one value
		if(verbosity>v_debug){
			std::cout<<"extracting config file json string from query result"<<std::endl;
		}
		std::string json_string = res[0][0].as<string>(); // res[0][0] is row 0, field 0.
		// XXX since we don't strictly return an error, a user can check if something went
		// wrong by checking if the returned string is empty. But in that case we need to
		// ensure a Tool that has no configuration files returns an empty json object,
		// not an empty string
		if(json_string=="") json_string = "{}"; // FIXME is this needed/appropriate?
		if(verbosity>v_debug){
			std::cout<<"returning json configfile string: "<<json_string<<"\n";
		}
		return json_string;
	}
	catch (const pqxx::syntax_error &e){
		std::cerr << e.what() << std::endl
		          << "When executing query: " << e.query()
		          << " at position " << e.error_position;
		if(e.sqlstate()!=""){
			std::cerr << ", with SQLSTATE error code: " << e.sqlstate();
		}
		std::cerr << std::endl;
	}
	catch (const pqxx::sql_error &e){
		std::cerr << e.what() << std::endl
		          << "When executing query: " << e.query();
		if(e.sqlstate()!=""){
			std::cerr << ", with SQLSTATE error code: " << e.sqlstate();
		}
		std::cerr<<std::endl;
	}
	catch (std::exception const &e){
		std::cerr << e.what() << std::endl;
	}
	return "";
}

int Postgres::InsertToolConfig(Store config, std::string toolname, std::string author, std::string description, std::string systemname){
	// insert a new Tool configuration entry
	try {
		// ensure we have an open connection to the database
		OpenConnection();
		
		// open a transaction to interact with the database
		//pqxx::work(*conn);
		if(verbosity>v_debug) std::cout<<"Opening nontransaction"<<std::endl;
		pqxx::nontransaction txn(*conn);
		
		// each entry in the configfiles table contains:
		// a tool name,          -
		// a version number,      |- together these 3 uniquely define a configuration file
		// a system name         -
		// an author
		// a creation timestamp
		// a description
		// the config file contents
		
		// if no systemname is given, we assume the user means the current system,
		// the name of which should be in the datamodel.
		if(systemname==""){
			get_ok = m_data->vars.Get("system",systemname);
			if(not get_ok){
				std::cerr<<"Postgres::InsertToolConfig error! no system name given and none in m_data!"<<std::endl;
				return false;
			}
		} else {
			// validate that this is a proper system name (we could drop the SELECT EXISTS (...) )
			std::string query_string = "SELECT EXISTS ( SELECT FROM pg_tables WHERE tablename = 'lappd' )";
			try {
				txn.exec1(query_string); // will throw if it doesn't return one entry
			} catch (pqxx::unexpected_rows &e){
				std::cerr<<"Postgres::InsertToolConfig Error! system name "<<systemname
				         <<" does not appear to be a valid system! Exception was: "<<std::endl
				         <<e.what();
				return -1;
			}
		}
		
		// Use the Store streamer to generate the json string
		std::string json_string;
		config >> json_string;
		
		// version number must be automatically assigned as the next unique version number
		// to find this out we need to query the database
		
		// query the latest version number for this tool
		std::string query_string = "SELECT max(version) FROM configfiles WHERE system = ";
		query_string += txn.quote(systemname) + " AND name = "+ txn.quote(toolname);
		if(verbosity>v_debug){
			std::cout<<"Querying max version number for tool "<<systemname<<"::"<<toolname<<std::endl;
		}
		pqxx::row row = txn.exec1(query_string);
		int versionnum=-1;
		if(row.size()>0){
			versionnum = row[0].as<int>();
		}
		if(verbosity>v_debug) std::cout<<"Max version number is "<<versionnum<<std::endl;
		++versionnum; // our new entry will be the next one up.
		
		// creation timestamp will also be automatically generated as NOW()
		std::string created="NOW()";
		
		// build the query
		query_string = 
		    "INSERT INTO configfiles ( name, version, system, author, created, description, contents) "
		    "VALUES ( $1, $2, $3, $4, $5, $6, $7 ) RETURNING id";
		pqxx::result res = txn.exec_params(query_string,
		                                   toolname,
		                                   versionnum,
		                                   systemname,
		                                   author,
		                                   created,
		                                   description,
		                                   json_string
		                                   );
		// important! commit the result
		//txn.commit();    // (unless we're use a nontransaction)
		int new_id = res[0][0].as<int>();
		
		// if no exceptions thrown, we're done.
		return versionnum;   // could also return the ID of the new row
	}
	catch (const pqxx::sql_error &e){
		std::cerr << e.what() << std::endl
			      << "When executing query: " << e.query();
		if(e.sqlstate()!=""){
			std::cerr << ", with SQLSTATE error code: " << e.sqlstate();
		}
		std::cerr<<std::endl;
		// from the discussion on the transactor framework page
		// (https://libpqxx.readthedocs.io/en/6.3/a00258.html)
		// it seems transactions can fail for transient reasons.
		// if for some reason we're not using the transactor framework
		// but still want to retry the query manually, do that here.
		// continue;    // along with any other necessary reinitializations and whatnot
	}
	catch (std::exception const &e){
		std::cerr << e.what() << std::endl;
	}
	// if we haven't returned true, something went wrong.
	return -1;
}

// XXX reminder that pqxx::result is a reference-counting wrapper and is not thread-safe! XXX
bool Postgres::Query(std::string query, int nret, pqxx::result* res, pqxx::row* row){
	// maybe this is redundant since OpenConnection will check is_open (against recommendations)
	for(int tries=0; tries<2; ++tries){
		// ensure we have a connection to work with
		if(OpenConnection()==nullptr){
			// no connection to batabase -> abort
			return false;
		}
		try{
			// open a transaction to interact with the database
			//pqxx::work(*conn);
			pqxx::nontransaction txn(*conn);
			
			// run the requested query
			// the type of exec we run is based on the user's expected number of returned rows, nret
			if(nret==0){
				txn.exec0(query);
			} else if(nret>0 && res==nullptr && row==nullptr){
				std::cerr<<"Postgres::ExecuteQuery called with expected number of returned rows "<<nret
				         <<" but nowhere to return the result!"<<std::endl;
				// we'll run the query anyway, just in case the user has some reason to invoke
				// a query with a return that they don't actually want, i guess....?
			} else if(nret>1 && res==nullptr){
				std::cerr<<"Postgres::ExecuteQuery called with expected number of returned rows "<<nret
				         <<" but only given a return pointer for one row! Only the first row will be returned!"
				         <<std::endl;
				// again we could be harsh and forbid this, which may help flag bugs, but we'll proceed...
			}
			if(nret==1 && res==nullptr && row!=nullptr){
				// user expects one returned row, and only wants the returned row. perfect.
				*row = txn.exec1(query);
			} else {
				// else either the user expects more than one row, or they want the pqxx::result,
				// so use a general exec
				if(res!=nullptr){
					*res = txn.exec(query);
					if(row!=nullptr){
						// i guess they want us to extract the first row too...?
						*row = (*res)[0];
					}
				} else {
					// they've only given us a row, but have told us they expect more than one row...
					pqxx::result loc_res = txn.exec(query);
					*row = loc_res[0];
				}
			}
			// if no exceptions thrown, we're done.
			return true;
		}
		catch (const pqxx::broken_connection &e){
			// if our connection is broken after all, disconnect, reconnect and retry
			if(tries==0){
				CloseConnection();
				delete conn; conn=nullptr;
				continue;
			}
		}
		catch (const pqxx::sql_error &e){
			std::cerr << e.what() << std::endl
				      << "When executing query: " << e.query();
			if(e.sqlstate()!=""){
				std::cerr << ", with SQLSTATE error code: " << e.sqlstate();
			}
			std::cerr<<std::endl;
			// from the discussion on the transactor framework page
			// (https://libpqxx.readthedocs.io/en/6.3/a00258.html)
			// it seems transactions can fail for transient reasons.
			// if for some reason we're not using the transactor framework
			// but still want to retry the query manually, do that here.
			// continue;    // along with any other necessary reinitializations and whatnot
		}
		catch (std::exception const &e){
			std::cerr << e.what() << std::endl;
		}
		break;   // if not explicitly 'continued', break.
	}
	// if we haven't returned true, something went wrong.
	return false;
}

