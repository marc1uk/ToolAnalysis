/* vim:set noexpandtab tabstop=4 wrap filetype=cpp */
#ifndef Postgres_H
#define Postgres_H

#include <string>
#include <iostream>
#include <sstream>
#include <pqxx/pqxx>
#include <sys/time.h>
#include <typeinfo>
#include <cxxabi.h>  // demangle

class DataModel;

class Postgres {
	
	public:
	Postgres(DataModel* m_data_in);
	~Postgres();
	void SetVerbosity(int verb);
	// open a connection to the database
	pqxx::connection* OpenConnection();
	// close the connection to the database. returns success, for whatever failure implies.
	bool CloseConnection();
	// get the json string representing a Tool's config variables
	std::string GetToolConfig(std::string toolname, int versionnum=-1, std::string systemname="");
	// get the toolchain configuration id for the given system based on the runconfig id and system name
	bool GetSystemConfig(bool update=true, int* systemconfig_in=nullptr, std::string systemname="", int runconfig=-1);
	// get the current run number and runconfig id
	bool GetCurrentRun(bool update=true, int* runnum_in=nullptr, int* runconfig_in=nullptr);
	
	// wrapper around exec since we handle the transaction and connection.
	// nret specifies the expected number of returned rows from the query.
	// res and row are outputs. return value is success.
	bool ExecuteQuery(std::string query, int nret, pqxx::result* res=nullptr, pqxx::row* row=nullptr);
	
	private:
	int verbosity=1;
	int v_error=0;
	int v_warning=1;
	int v_message=2;
	int v_debug=3;
	std::string logmessage;
	int get_ok;
	DataModel* m_data=nullptr;
	pqxx::connection* conn=nullptr;
	
	// default connection details
	std::string dbname="postgres";
	std::string hostaddr="127.0.0.1";
	std::string hostname="";
	int port=5432;
	std::string dbuser="postgres";
	std::string password="pass";
	
	
	public:
	
//	template <typename Tuple>
//	bool ExecuteQuery(std::string query_string, std::vector<Tuple>& rets){
//		// run an SQL query and try to return the results
//		// into a vector of tuples, one entry per row.
//		// tuple contents must be compatible with the returned columns.
//		pqxx::result local_ret;
//		bool success = ExecuteQuery(query_string, 2, &local_ret, nullptr);
//		if(not success) return false; // query failed
//		
//		// XXX
//	}
	
	template <typename... Ts>
	bool ExecuteQuery(std::string query_string, Ts&&... rets){
		// run an SQL query and try to pass the results
		// into a parameter pack. the passed arguments
		// must be compatible with the returned columns
		pqxx::row local_row;
		bool success = ExecuteQuery(query_string, 1, nullptr, &local_row);
		if(not success) return false; // query failed
		
		success = ExpandRow<sizeof...(Ts), Ts...>::expand(local_row, std::forward<Ts...>(rets...));
	}
	
	////////
	// helper function to use the contents of a pqxx::row
	// to populate a parameter pack
	template<std::size_t N, typename T, typename... Ts>
	struct ExpandRow {
		static bool expand(const pqxx::row& row, T last, Ts&... out){
			bool ok = ExpandRow<N-1, Ts...>::expand(row, out...);
			if(not ok) return false; // do not attempt further expansion.... i mean, we could...?
			try{
				row[pqxx::row::size_type(N-1)].to(last);
			}
			catch (const pqxx::sql_error &e){
				std::cerr << e.what() << std::endl
						  << "When executing query: " << e.query();
				if(e.sqlstate()!=""){
					std::cerr << ", with SQLSTATE error code: " << e.sqlstate();
				}
				std::cerr<<std::endl;
				std::cerr<<"Postgres::ExpandRow failed to convert sql return field 0 to output type "
				         <<abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, nullptr)
				         <<std::endl;
				return false;
			}
			catch (std::exception const &e){
				std::cerr << e.what() << std::endl;
				std::cerr<<"Postgres::ExpandRow failed to convert sql return field "<<(N-1)<<" to output type "
				         <<abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, nullptr)
				         <<std::endl;
				return false;
			}
		}
	};
	
	template<typename T>
	struct ExpandRow<1, T> {
		static bool expand(const pqxx::row& row, T& out){
			try{
				row[0].to(out);
			}
			catch (const pqxx::sql_error &e){
				std::cerr << e.what() << std::endl
						  << "When executing query: " << e.query();
				if(e.sqlstate()!=""){
					std::cerr << ", with SQLSTATE error code: " << e.sqlstate();
				}
				std::cerr<<std::endl;
				std::cerr<<"Postgres::ExpandRow failed to convert sql return field 0 to output type "
				         <<abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, nullptr)
				         <<std::endl;
				return false;
			}
			catch (...){
			}
			return true;
		}
	};
	// end helper function
	////////

};


#endif
