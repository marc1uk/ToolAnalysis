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

class Postgres {
	
	public:
	Postgres();
	~Postgres();
	void Init(std::string hostname_in="", std::string hostip_in="", int port_in=-1,
	          std::string user_in="", std::string password_in="", std::string dbname_in="");
	
	void SetVerbosity(int verb);
	// open a connection to the database
	pqxx::connection* OpenConnection(std::string* err=nullptr);
	// close the connection to the database. returns success, for whatever failure implies.
	bool CloseConnection(std::string* err=nullptr);
	
	// wrapper around exec since we handle the transaction and connection.
	// nret specifies the expected number of returned rows from the query.
	// res and row are outputs. return value is success.
	bool Query(std::string query, int nret, pqxx::result* res=nullptr, pqxx::row* row=nullptr, std::string* err=nullptr);
	
	// one that returns the value of many fields from one row as strings in a vector (row_or_col='r')
	// or the value of one field from many rows (row_or_col='c')
	bool QueryAsStrings(std::string query, std::vector<std::string> *results, char row_or_col, std::string* err=nullptr);
	
	// one that returns rows as json strings representing maps of fieldname:value
	// we uhhhhh need to check this works ok with nested strings and stuff
	bool QueryAsJsons(std::string query, std::vector<std::string> *results, std::string* err=nullptr);
	
	private:
	int verbosity=1;
	int v_error=0;
	int v_warning=1;
	int v_message=2;
	int v_debug=3;
	std::string logmessage;
	int get_ok;
	pqxx::connection* conn=nullptr;
	
	// default connection details
	std::string dbname="";
	std::string hostaddr="";
	std::string hostname="";
	int port=-1;
	std::string dbuser="";
	std::string dbpasswd="";
	
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
		bool success = Query(query_string, 1, nullptr, &local_row);
		if(not success) return false; // query failed
		
		success = ExpandRow<sizeof...(Ts), Ts&&...>::expand(local_row, std::forward<Ts>(rets)...);
		return success;
	}
	
	////////
	// helper function to use the contents of a pqxx::row
	// to populate a parameter pack
	template<std::size_t N, typename T, typename... Ts>
	struct ExpandRow {
		static bool expand(const pqxx::row& row, T& last, Ts&... out){
			//std::cout << __PRETTY_FUNCTION__ << "\n";
			bool ok = ExpandRow<N-1, Ts...>::expand(row, std::forward<Ts>(out)...);
			if(not ok) return false; // do not attempt further expansion.... i mean, we could...?
			try{
			//	std::cout<<"extracting argument "<<(row.size()-pqxx::row::size_type(N))<<" to variable of type "
			//	         <<type_name<decltype(last)>()<<std::endl;
				row[row.size()-pqxx::row::size_type(N)].to(last);
			}
			catch (const pqxx::sql_error &e){
				std::cerr << e.what() << std::endl
						  << "When executing query: " << e.query();
				if(e.sqlstate()!=""){
					std::cerr << ", with SQLSTATE error code: " << e.sqlstate();
				}
				std::cerr<<std::endl;
				std::cerr<<"Postgres::ExpandRow failed to convert sql return field "
				         <<(row.size()-pqxx::row::size_type(N))<<" to output type "
				         <<typeid(last).name()<<std::endl;
				         //<<type_name<decltype(last)>()<<std::endl;
				return false;
			}
			catch (std::exception const &e){
				std::cerr << e.what() << std::endl;
				std::cerr<<"Postgres::ExpandRow failed to convert sql return field "
				         <<(row.size()-pqxx::row::size_type(N))<<" to output type "
				         <<typeid(last).name()<<std::endl;
				         //<<type_name<decltype(last)>()<<std::endl;
				return false;
			}
		}
	};
	
	template<typename T>
	struct ExpandRow<1, T> {
		static bool expand(const pqxx::row& row, T& out){
			//std::cout << __PRETTY_FUNCTION__ << "\n";
			try{
				//std::cout<<"extracting last argument "<<(row.size()-1)<<" to variable of type "
				//         <<type_name<decltype(out)>()<<std::endl;
				row[row.size()-1].to(out);
			}
			catch (const pqxx::sql_error &e){
				std::cerr << e.what() << std::endl
						  << "When executing query: " << e.query();
				if(e.sqlstate()!=""){
					std::cerr << ", with SQLSTATE error code: " << e.sqlstate();
				}
				std::cerr<<std::endl;
				std::cerr<<"Postgres::ExpandRow failed to convert sql return field "<<(row.size()-1)
				         <<" to output type "<<typeid(out).name()<<std::endl;
				         //<<" to output type "<<type_name<decltype(out)>()<<std::endl;
				return false;
			}
			catch(std::exception const &e){
				std::cerr << e.what() << std::endl;
				std::cerr<<"Postgres::ExpandRow failed to convert sql return field "
				         <<(row.size()-1)<<" to output type "
				         <<typeid(out).name()<<std::endl;
				         //<<type_name<decltype(out)>()<<std::endl;
				return false;
			}
			return true;
		}
	};
	// end helper function
	////////
	
	
	////////
	// helper function for insertions
	template <typename... Rest>
	bool Insert(std::string tablename, std::vector<std::string> &fields, std::string* err, Rest... args){
		// maybe this is redundant since OpenConnection will check is_open (against recommendations)
		for(int tries=0; tries<2; ++tries){
			// ensure we have a connection to work with
			if(OpenConnection(err)==nullptr){
				// no connection to batabase -> abort
				return false;
			}
			try{
				// open a transaction to interact with the database
				//pqxx::work(*conn);
				pqxx::nontransaction txn(*conn);
				
				// form the query
				std::string query_string = "INSERT INTO " + tablename + "( ";
				for(auto&& afield : fields) query_string += afield +", ";
				query_string = query_string.substr(0, query_string.length()-2); // pop trailing comma
				query_string += ") VALUES (";
				for(int i=0; i<fields.size(); ++i) query_string += "$" + std::to_string(i+1) + ", ";
				query_string = query_string.substr(0, query_string.length()-2); // pop trailing comma
				query_string += ")";
				
				// try to run it
				txn.exec_params(query_string, args...);
				
				// don't expect (or handle) a return.
				// if we haven't thrown an exception, we're done.
				return true;
			}
			catch (const pqxx::broken_connection &e){
				// if our connection is broken after all, disconnect, reconnect and retry
				if(tries==0){
					CloseConnection();
					delete conn; conn=nullptr;
					continue;
				} else {
					std::cerr<<"Postgres::Query error - broken connection, failed to re-establish it"<<std::endl;
					if(err) *err=e.what();
				}
			}
			catch (const pqxx::sql_error &e){
				std::string msg = e.what();
					        msg += "When executing query: " + e.query();
				if(e.sqlstate()!=""){
					msg += ", with SQLSTATE error code: " + e.sqlstate();
				}
				std::cerr<<msg<<std::endl;
				if(err) *err = msg;
			}
			catch (std::exception const &e){
				std::cerr << e.what() << std::endl;
				if(err) *err = e.what();
			}
			break;   // if not explicitly 'continued', break.
		} // end tries
		return false;
	}
	// end helper function
	////////
};


#endif
