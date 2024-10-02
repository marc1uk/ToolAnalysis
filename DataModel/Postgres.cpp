/* vim:set noexpandtab tabstop=4 wrap filetype=cpp */
#include "Postgres.h"
//#include <signal.h>
#include <unistd.h>   // fork
#include <stdlib.h>   // system
#include <sys/wait.h> // waitpid
#include <fstream>

void Postgres::SetVerbosity(int verb){
	verbosity=verb;
}

// TODO probably could do better with the exception handling throughout this file
pqxx::connection* Postgres::OpenConnection(std::string* err){
	if(verbosity>v_debug) std::cout<<"Opening Connection"<<std::endl;
	try{
		// if we already have a connection open, nothing to do
		if(conn!=nullptr && conn->is_open()){
			if(verbosity>v_debug) std::cout<<"Connection already open"<<std::endl;
			return conn;
		} else if(conn){
			// conn not null, but is not open...
			delete conn; // should we do this? e.g. lazy connections will open only on first use.
		}
		
		// otherwise form the connection string
		std::stringstream tmp;
		if(dbname!="")   tmp<<" dbname="<<dbname;
		if(port!=-1)     tmp<<" port="<<port;
		if(dbuser!="")   tmp<<" user="<<dbuser;
		if(dbpasswd!="") tmp<<" password="<<dbpasswd;
		if(hostaddr!="") tmp<<" hostaddr="<<hostaddr;
		if(hostname!="") tmp<<" host="<<hostname;
		
		// attempt to connect to the database
		if(verbosity>v_debug) std::cout<<"connecting with string '"<<tmp.str()<<"'"<<std::endl;
		conn = new pqxx::connection(tmp.str().c_str());
		
		// verify we succeeded
		// "don't use is_open(), use the broken_connection exception", they say. Hmm.
		// but will that be thrown now, or only when we try to *use* the connection, for a transaction?
		if(!conn->is_open()){
			std::cerr<<"Failed to connect to the database! Connection string was: '"
			         <<tmp.str()<<"', please verify connection details"<<std::endl;
			if(err) *err = "pqxx::connection::is_open() returned false after connection attempt";
			return nullptr;
		}
		return conn;
	}
	catch (const pqxx::broken_connection &e){
		// as usual the doxygen sucks, but it seems this doesn't provide
		// any further methods to obtain information about the failure mode,
		// so probably not useful to catch this explicitly.
		std::cerr << e.what() << std::endl;
		if(err) *err = e.what();
	}
	catch (std::exception const &e){
		std::cerr << e.what() << std::endl;
		if(err) *err = e.what();
	}
	return nullptr;
}

bool Postgres::CloseConnection(std::string* err){
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
			if(err) *err="pqxx::connection::is_open() returns true even after calling disconnect";
			return false;
		}
		return true;
	}
	catch (const pqxx::broken_connection &e){
		return true; // sure
	}
	catch (std::exception const &e){
		std::cerr << e.what() << std::endl;
		if(err) *err=e.what();
		return false; // umm....
	}
	return false; // dummy
}

Postgres::~Postgres(){
	CloseConnection();
	if(conn) delete conn;
}

Postgres::Postgres(){}

void Postgres::Init(std::string hostname_in, std::string hostip_in, int port_in,
               std::string user_in, std::string password_in, std::string dbname_in){
	// apparently when a connection breaks, it not only throws an exception
	// but sends a SIGPIPE signal that by default will kill the application!
	// https://libpqxx.readthedocs.io/en/6.3/a00915.html
	// let's please not do that.
	signal(SIGPIPE, SIG_IGN);
	// set connection details
	hostname=hostname_in;
	hostaddr=hostip_in;
	port=port_in;
	dbuser=user_in;
	dbname=dbname_in;
	dbpasswd=password_in;
}

// XXX reminder that pqxx::result is a reference-counting wrapper and is not thread-safe! XXX
bool Postgres::Query(std::string query, int nret, pqxx::result* res, pqxx::row* row, std::string* err){
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
			
			// run the requested query
			// the type of exec we run is based on the user's expected number of returned rows, nret
			if(nret==0){
				txn.exec0(query);
			} else if(res==nullptr && row==nullptr){
				std::string msg =  "Postgres::ExecuteQuery called with expected number of returned rows ";
				            msg += std::to_string(nret)+" but nowhere to return the result!";
				std::cerr<<msg<<std::endl;
				if(err) *err=msg;
				// we'll run the query anyway, just in case the user has some reason to invoke
				// a query with a return that they don't actually want, i guess....?
			} else if(nret>1 && res==nullptr){
				std::string msg =  "Postgres::ExecuteQuery called with expected number of returned rows ";
				            msg += std::to_string(nret)+" but only given a return pointer for one row!";
				            msg += "Only the first row will be returned";
				std::cerr<<msg<<std::endl;
				if(err) *err=msg;
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
			// from the discussion on the transactor framework page
			// (https://libpqxx.readthedocs.io/en/6.3/a00258.html)
			// it seems transactions can fail for transient reasons.
			// if for some reason we're not using the transactor framework
			// but still want to retry the query manually, do that here.
			// continue;    // along with any other necessary reinitializations and whatnot
		}
		catch (std::exception const &e){
			std::cerr << e.what() << std::endl;
			if(err) *err = e.what();
		}
		break;   // if not explicitly 'continued', break.
	}
	// if we haven't returned true, something went wrong.
	return false;
}

bool Postgres::QueryAsStrings(std::string query, std::vector<std::string> *results, char row_or_col, std::string* err){
	// generically run a query, without knowing how many returns are expected.
	// we'll need to get the results in a generic pqxx::result, and specify the number
	// of returned rows is >1. If there's fewer, it'll just return an empty container.
	pqxx::result res;
	get_ok = Query(query, 2, &res, nullptr, err);
	// if the query failed, the user didn't provide means for a return, or the query had no return,
	// then we have no need to parse the response and we're done.
	if(not get_ok || results==nullptr || res.size()==0) return get_ok;
	// otherwise, parse the response
	// we're given a vector for putting results in.
	// this may be several fields of one row, or one field from several rows
	if(row_or_col=='r'){
		// row mode: user is querying many fields from one row
		pqxx::row row = res[0];  // XXX we discard any notice of additional rows...
		// Iterate over fields
		for (const pqxx::field field : row){
			results->push_back(field.c_str());
		}
	} else {
		// column mode: one column from many rows
		for(const pqxx::row row : res){
			pqxx::field field = row[0];  // XXX we discard any notice of additional fields...
			results->push_back(field.c_str());
		}
	}
	return true;
}

bool Postgres::QueryAsJsons(std::string query, std::vector<std::string> *results, std::string* err){
	// generically run a query, without knowing how many returns are expected.
	// we'll need to get the results in a generic pqxx::result, and specify the number
	// of returned rows is >1. If there's fewer, it'll just return an empty container.
	//printf("QueryAsJsons running '%s'\n",query.c_str());
	pqxx::result res;
	get_ok = Query(query, 2, &res, nullptr, err);
	// if the query failed, the user didn't provide means for a return, or the query had no return,
	// then we have no need to parse the response and we're done.
	if(not get_ok || results==nullptr || res.size()==0) return get_ok;
	// otherwise, parse the response. iterate over returned rows
	for(pqxx::row row : res){
		// build a json from fields in this row
		std::stringstream tmp;
		tmp << "{";
		for (pqxx::row::iterator it=row.begin(); it<row.end(); ){
			tmp << "\"" << it->name() << "\":\""<< it->c_str() << "\"";
			++it;
			if(it!=row.end()) tmp << ", ";
		}
		tmp << "}";
		results->push_back(tmp.str());
	}
	return true;
}

