/* vim:set noexpandtab tabstop=4 wrap filetype=cpp */
#include "PGTool.h"
#include <typeinfo>
#include <cxxabi.h>  // demangle

PGTool::PGTool():Tool(){
	// get the name of the tool from its class name
	toolName=abi::__cxa_demangle(typeid(*this).name(), nullptr, nullptr, nullptr);
}

// List of relations
// Schema |    Name     | Type  |  Owner   |    Size    | Description 
//--------+-------------+-------+----------+------------+-------------
// public | configfiles | table | postgres | 8192 bytes | 
// public | lappd       | table | postgres | 8192 bytes | 
// public | run         | table | postgres | 8192 bytes | 
// public | runconfig   | table | postgres | 8192 bytes | 

/*

psql -c "INSERT INTO run (runnum, subrunnum, start, stop, runconfig, notes) VALUES(0, 0, timestamp '2020-09-16 15:54:00', timestamp '2020-09-16 17:33:00', 0, 'dummy run')"

// turns out auto-increment default keys start from 1 not 0, so above 'runconfig' id of 0 doesn't match anything.
psql -c "INSERT INTO run (runnum, subrunnum, start, stop, runconfig, notes) VALUES(1, 0, timestamp '2021-09-17 14:13:00', timestamp '2021-09-17 14:14:00', 1, 'dummy run v2')"

psql -c "INSERT INTO runconfig ( name, created, geometry, daq, vme, hv, lappd, mrd, description, author) VALUES( 'dummyrun', timestamp '2020-09-16 15:54:00', 0, 0, 0, 0, 0, 0, 'dummy toolchain configuration', 'moflaher')"

psql -c "INSERT INTO lappd (created, description, configfiles) VALUES ( timestamp '2020-09-16 15:54:00', 'test toolchain 1', '{\"PGTool\":0, \"OtherTool\":6}'::jsonb )"

psql -c "INSERT INTO configfiles (system, name, version, description, created, author, contents) VALUES ( 'lappd', 'PGTool', 0, 'PGTool config file v0', timestamp '2020-09-16 15:54:00', 'moflaher', '{ \"verbosity\":1, \"somestring\":\"here is a string variable\", \"adouble\":3.14, \"aboolean\":true }'::jsonb )"

// n.b. type specifiers such as "'YYYY-MM-DD HH:MM:SS'::timestamp" or "{ \"some\":json }::jsonb"
// are needed for command line, but are not correct for parametrized queries, which seem
// to be the only reliable way to insert data.
*/

bool PGTool::Initialise(std::string configfile, DataModel &data){
	
	m_data= &data; //assigning transient data pointer
	m_data->vars.Set("system","lappd");   // for DAQ systems this would be in the datamodel constructor
	
	// Retrieve varibles from the postgres database
	std::cout<<toolName<<": Getting config"<<std::endl;
	m_data->postgres.SetVerbosity(10);
	std::string json_data = m_data->postgres.GetToolConfig(toolName);
	std::cout<<"JSON version of Tool config was: "<<json_data<<std::endl;
	if(json_data==""){
		Log(toolName+" Error! Failed to get Tool config from database!",v_error,verbosity);
		return false;
	}
	
	// parse the json string to populate the usual m_variables Store.
	//if(configfile!="") m_variables.Initialise(configfile); // old method, populate from a local file
	if(json_data!="") m_variables.JsonParser(json_data);     // new method, populate from json string
	
	// that's it. Print contents to check.
	m_variables.Print();
	
	// let's now try to run a general database query.
	// The postgres interface class is used to execute the query, and returns the results directly
	// into end-user variables: `m_data->postgres.ExecuteQuery(query, output1, output2...);`
	// Any number of outputs can be given, of arbitrary types, but they must of course match what
	// will be returned from the query.
	// For now, this only supports queries that return just one row.
	// TODO extend this interface to support multiple returned rows via vectors or something.
	std::string query_string = "SELECT id, name, contents FROM configfiles WHERE system = 'lappd'";
	// variables for the outputs. Types should be appropriate.
	int out_id;
	std::string out_name;
	std::string out_json;
	std::cout<<"performing ExecuteQuery"<<std::endl;
	// run the query. variable order should match that SELECTed in the query.
	m_data->postgres.ExecuteQuery(query_string, out_id, out_name, out_json);
	// print out the results to validate
	std::cout<<"variadic query done, ID was "<<out_id<<", out name was "<<out_name
	         <<", out_json was "<<out_json<<std::endl;
	
	// For inserting new configuration files we can do the following.
	// simply define a normal Store
	Store varstore;
	// populate with the desired variables
	int anintvar = 99;                     varstore.Set("myint",anintvar);
	double adoublevar = 33.32;             varstore.Set("mydouble",adoublevar);
	std::string astringvar = "oh hello";   varstore.Set("mystring",astringvar);
	// and pass it to the Postgres interface class with all the required metadata
	std::string thetool = "MyTotallyCoolTool";         // this is your tool's class name
	std::string systemname  = "lappd";                 // ensure this is a valid system name
	std::string authorname = "billybob";               // who wrote this config file
	std::string description = "increase buffer size";  // changes made and why
	// we won't run the command since it would pollute the database with a dummy entry.
	/*
	int versionnum = m_data->postgres.InsertToolConfig(varstore, thetool, authorname, description, systemname);
	*/
	// the return will be the version number of the newly created config file entry.
	// this is automatically assigned as the next free version number
	// for a configfile entry with the specified system name and tool name.
	
	return true;
}

bool PGTool::Execute(){
	
	// below are some notes and tests on using the underlying libpqxx.
	// there shouldn't be any reason users will need to interface with libpqxx
	// directly, they can use the interface class from the DataModel.
	
	// pqxx doesn't return error codes to indicate failure, it throws exceptions,
	// so always put libpqxx code within a try/catch block
	try {
		// open the connection to the database
		/*
		// XXX this should be forbidden - opening a new connection will force close
		// any existing connections, such as the one held by the DataModel postgres class.
//		std::string connectionstring = "postgresql://postgres@localhost/testb";
//		std::string connectionstring = "hostaddr=127.0.0.1 port=5433 dbname=postgres user=postgres password=pass";
		std::string connectionstring = "host=/tmp port=5434 user=postgres password=pass";
		std::cout<<"Opening connection"<<std::endl;
		pqxx::connection myConnection{connectionstring};
		*/
		// XXX Please only use the following
		std::cout<<"Opening connection"<<std::endl;
		pqxx::connection* myConnection = m_data->postgres.OpenConnection();
		
		std::cout<<"Making transaction"<<std::endl;
		pqxx::work myTransaction{*myConnection};
		// or e.g. pqxx::nontransaction myTransaction{myConnection}.
		// pqxx::work is a wrapper around the transaction base class.
		// it's the suggested default. Not sure why. Unlike pqxx::nontransaction,
		// any INSERT/UPDATE/DELETE calls made via a pqxx::work must be followed
		// by a pqxx::work::commit() call or the changes will be rolled back!
		// note that calling pqxx::transaction::commit() closes the transaction so it cannot
		// be re-used for subsequent queries!
		
		// these are the contents of the configfiles table, for now.
		std::string systemname = "lappd";                          // 'system'      - text field
		std::string toolname = toolName;                           // 'name'        - text field
		int version=0;                                             // 'version'     - integer field
		std::string author = "moflaher";                           // 'author'      - text field
		
		// to uniquely identify a particular tool's config variables we need to specify:
		// 1. the tool name
		// 2. the system name (e.g. DAQ/VME/MRD...). This represents which toolchain
		//    the tool refers to, permitting tools by the same name on each different system.
		// 3. the config file version number.
		std::string query_string = "SELECT contents FROM configfiles WHERE name ="
		   + myTransaction.quote(toolname) + " AND system = " + myTransaction.quote(systemname)
		   +" AND version = " + pqxx::to_string(version);
		// XXX note the use of txn.quote to add in proper escaping whenever queries involve string variables.
		// Use 'txn::quote(...) for string variables, and 'txn::quote_name(...) for column or table names.
		// XXX note also the use of pqxx::to_string instead of std::to_string. This should be used
		// to insert basic datatypes into query strings. It forms the appropriate string POSTGRES
		// will use to store your datatype, which apparently has its own particular format.
		// (see https://libpqxx.readthedocs.io/en/6.3/a00256.html ) 
		std::cout<<"Running the following query: \n" <<query_string <<"\n";
		
		// general queries that return multiple multiple rows must use `pqxx::transaction::exec(query_string)`
		// which returns a pqxx::result that encapsulates all returned rows, each in a pqxx::row object.
		// for queries that return only one row, we can use `pqxx::transaction::exec1(...)` to bypass
		// the intermediate pqxx::result and directly access the pqxx::row. This will throw an exception
		// if the query matches more than one entry. each pqxx::row encapsulates potentially many pqxx::fields.
		// XXX NOTE the pqxx::result class is implemented as a reference-counting smart pointer XXX
		// XXX which means pqxx::result objects are not thread-safe!                            XXX
		/*
		pqxx::result results = myTransaction.exec(query_string);
		pqxx::row row = results[0];
		*/
		// or
		pqxx::row row = myTransaction.exec1(query_string);
		std::cout<<"query returned "<<row.size()<<" columns"<<std::endl;
		
		// to retrieve a field contents into a variable you need to specify the desired c++ datatype
		// an exception will be thrown if the stored field type is not compatible with the requested
		// c++ type.
		// config files are stored as a jsonb datatype, but they may be transparently
		// converted to string
		std::string json_data = row[0].as<string>();
		std::cout<<"json data was "<<json_data<<std::endl;
		
		// for queries that return no value, such as `UPDATE` or `INSERT` commands, we can use
		// myTransaction.exec0(query_string), which will throw an exception if anything is returned.
		
		// for any changes to the database to persist (such as UPDATE or INSERT) we must commit the result via
		// myTransaction.commit() - with the exception of ppxx::nontransaction exec calls, which automatically
		// commit the result immediately. It is, however, good practice to call 'commit' anyway,
		// to build good habits for when using other transactional types. Note you can call 'commit'
		// even if no changes are made. If the transaction threw an exception, it will have aborted
		// --- and been destroyed (XXX is that true or just because of the scope of the transaction?),
		// so you should wrap the exec() call to prevent calling commit if the query failed.
		// If something goes wrong and the query changes should be forgotten, call abort() instead.
		// If you do neither, an implicit abort() is executed at destruction time.
		//std::cout<<"committing the transaction (noop)"<<std::endl;
		//myTransaction.commit();  // this closes the transaction, so we can no longer use it!
		
		// for a more robust way to perform transactions, consider the transactor framework:
		// https://libpqxx.readthedocs.io/en/6.3/a00258.html
		// or the pqxx::robusttransaction, instead of pqxx::work or pqxx::nontransaction
		// https://libpqxx.readthedocs.io/en/6.3/a01147.html
		
		// returned result objects are independent of the transaction and connection.
		// a general result may have many rows and columns, so we need to iterate over them
		std::cout<<"run a query and this time obtain a pqxx::result"<<std::endl;
		query_string = "SELECT * FROM configfiles WHERE name =" + myTransaction.quote(toolname)
		             + " AND system = " + myTransaction.quote(systemname);
		pqxx::result results = myTransaction.exec(query_string);
		for(const pqxx::row row : results){
			std::cout << "Next Result Row: "<<std::endl;
			// Iterate over fields in a row
			for (const pqxx::field field: row){
				// XXX All fields are internally stored as strings, so we can invoke .c_str() on ANY data type
				// XXX BUT this will escape non-standard characters, and return up to the first 'terminating null',
				// XXX which may result in mangled or truncated results in the case of strings or binary data XXX
				// ^^^ is this true?
				std::cout << "> field "<< field.name() <<" had value " << field.c_str() << std::endl;
			}
			// we can also directly access fields by name...
			std::cout<<"creationdate was "<<row["created"].c_str()<<std::endl;
			// ... or by index. we can parse directly into the correct type by field.as<type>()
			std::cout<<"created by user "<<row[6].as<string>()<<std::endl;
			// ... or use the result to directly set the value of an object by field.to(destination_object);
			std::cout<<"returned columns and their properties are:"<<std::endl;
			// n.b. we can of course also use iterators...
			for (auto field = row.begin(); field != row.end(); field++){
				// ... but what's unique about pqxx iterators is that they can be used directly
				// as if they were rows or fields, without dereferencing. So we can do this:
				std::cout << field.name() << ", "  // as well as (*field).name() or field->name()
				          << field->table() << ", "
				          << (*field).type() << std::endl;
				// note the above properties all return OIDs, so they're not directly meaningful... TODO
			}
		}
		
		// timestamp fields
		// ----------------
		// timestamps can be specified using an appropriately formatted string: YYYY-MM-DD HH:MM:SS,
		// or the current date+time can be specified by passing the string "NOW()"
		std::string timestamp = "2020-09-16 20:05:00";                   // 'created' - timestamp field
		// TODO discuss potential formats including use of c++ date/time utils
		
		// bytea fields
		// ------------
		// 'bytea' fields can store arbitrary binary data, but in our case we're mostly using
		// them to store simple text strings of arbitrary length, such as in the description field.
		// To write text data to bytea fields just using the normal quoting - myTransaction.quote(std::string).
		std::string description = "Here's a dummy config file";         // 'description' - bytea field
		std::string jsoncontents = "{ \"PGTool\":2 }";                  // 'contents'    - jsonb field*
		// *see below for discussion of json field handling
		
		// n.b. the most reliable to achieve insertion without needing to worry about
		// suitable quoting or escaping is to use a parameterized query. This is done by specifying
		// placeholders $1, $2 ... in the query, and then using 'pqxx::transaction::exec_params'
		// with the actual variables holding the data passed as additional arguments:
		query_string = "INSERT INTO lappd (created, description, configfiles) VALUES ( $1, $2, $3 )";
		std::cout<<"Testing writing bytea fields with query: \n"<<query_string<<"\n";
		myTransaction.exec_params(query_string,
		                         timestamp,
		                         description,
		                         jsoncontents
		                         );
		//myTransaction.commit();
		
		// When retrieving bytea fields holding textual data, we need to convert the stored binary
		// back into text, or else we'll end up with gibberish.
		// Perhaps unintuitively we do this by *encoding* (not decoding!) the field to 'escape' type.
		// https://www.postgresql.org/docs/9.4/functions-binarystring.html
		// https://www.postgresql.org/docs/9.3/functions-string.html
		// n.b. since the data returned is the output of 'encode' (and not the raw field data)
		// the queried field name isn't preserved, but we can restore it by aliasing with 'AS'
		query_string = "SELECT encode(description, 'escape')::text AS description FROM lappd";
		// we can also use 'convert_from' and specify the locale, e.g. UTF-8
		query_string = "SELECT convert_from(description, 'UTF-8') AS description FROM lappd";
		
		/*
		e.g.  (specifying '::bytea' after 'description' is optional)
		"SELECT id, created, encode(description::bytea, 'escape') as description, configfiles from lappd"
		returns the requested row with the description field translated
		+----------------------------------------------------+
		| id          | 1                                    |
		| created     | 2020-09-16 15:54:00                  |
		| description | test toolchain 1                     |
		| configfiles | {"PGTool": 0, "OtherTool": 6}        |
		+----------------------------------------------------+
		*/
		
		// Alternatively we can do the conversion client-side with a pqxx::binarystring.
		// we build a pqxx::binarystring from the returned pqxx::field
		// XXX NOTE pqxx::binarystring is implemented as a reference-counting smart pointer XXX
		// XXX which means pqxx::binarystring objects are not thread-safe!                  XXX
		query_string = "SELECT description FROM configfiles WHERE name = "+myTransaction.quote(toolname)
		             + " AND system = " + myTransaction.quote(systemname) 
		             + " AND version = " + pqxx::to_string(version);
		std::cout<<"testing reading bytea fields with query: \n"<<query_string<<"\n";
		results = myTransaction.exec(query_string);
		pqxx::field byteafield = results[0][0];
		pqxx::binarystring my_bytea_fielddata = pqxx::binarystring(byteafield);
		// we then do the unscrambling with the pqxx::binarystring::str() method:
		// n.b. the .str() method appends a trailing '\0' to the data, which is necessary to form a properly
		// terminated std::string, so be sure to use this and not '.get()' unless you also use '.length()'!
		std::string unscrambled = my_bytea_fielddata.str();
		std::cout <<"scrambled description is "<< byteafield.as<string>() <<std::endl;
		std::cout <<"unscrambled description is '"<< unscrambled <<"'"<<std::endl;
		
		/*
		// N.B. we can avoid having to explicitly convert every time by creating a VIEW:
		// This modifies the database structure, so ask your friendly db admin before doing this!
		psql -c "CREATE VIEW lappd_view AS " \
		        " SELECT id, created, encode(description::bytea, 'escape') as description, configfiles " \
		        "FROM lappd; "
		// then querying that view like a table - "SELECT * from lappd_view" - will return
		// rows with the description column already converted back to text.
		*/
		
		// jsonb fields
		// ------------
		// configfiles are imported to and exported from Stores as json strings.
		// postgres has a field format specifically for json data, which ensures syntatic validity
		// on record entry, and allows direct access to elements within stored json fields.
		// ( see https://www.postgresql.org/docs/current/datatype-json.html for details)
		jsoncontents = "{ \"bar\":\"baz\", \"balance\":7.77, \"active\":false }";
		// we saw this used for inserting data above.
		
		// unlike bytea fields, data retrieved from jsonb fields comes straight back out
		// as a plain json string, so no further work is needed. But we do have other options!
		
		// we can access json object map contents directly by using 'jsonb_each(fieldname)'
		// we first SELECT the jsonb field, and by then calling "jsonb_each(...)" we can convert
		// the returned json string into to a temporary table from which we may SELECT specific keys
		query_string = "SELECT * FROM jsonb_each((SELECT contents FROM configfiles))";
		/* returns e.g. (assuming we selected a 'contents' entry with a json holding these contents)
		+-------------------------------------------+
		|    key     |            value             |
		| -----------+----------------------------- |
		| adouble    | 3.14                         |
		| aboolean   | true                         |
		| verbosity  | 1                            |
		| somestring | "here is a string variable"  |
		+-------------------------------------------+
		// we could of course have used "SELECT verbosity FROM ..." to retrieve just the verbosity.
		*/
		
		// another alternative to directly access keys within a json is via json_field->'keyname'
		// So to access the verbosity we could also have used:
		query_string = "select contents->'verbosity' AS \"verbosity\" FROM configfiles";
		// n.b. The returned field will show up with field name "?column?" unless with alias it with 'AS'
		
		// we can even use this within query selection criteria - but note that
		// this requires type casting the selected object from json to the correct type.
		query_string = "SELECT * FROM lappd WHERE (configfiles->'OtherTool')::integer = 6";
		// Find users where preferences.beta is true
		query_string = "SELECT preferences->'beta' FROM users WHERE (preferences->>'beta')::boolean is true";
		
		// n.b. The short arrow -> keeps the type as JSON, and the long arrow ->> converts to text
		// so the following are both the same:
		/*
		psql -c "SELECT * FROM configfiles WHERE contents->'somestring' = '\"here is a string variable\"' "
		// or
		psql -c "SELECT * FROM configfiles WHERE contents->>'somestring' = 'here is a string variable' "
		*/
		
		// we can also 'SELECT * FROM json_object_keys(json_field)' to retrieve a list of stored keys.
		query_string = "SELECT * FROM json_object_keys((SELECT contents FROM configfiles)::json)";
		// and many more! for other json direct-access functions see:
		// https://www.postgresql.org/docs/current/functions-json.html
		// https://devhints.io/postgresql-json
		
		// n.b. while json allows arrays and objects to be nested arbitrarily as shown below,
		// but this is not supported by the Store::JsonParser at this time, so please avoid this!
		jsoncontents = "{ \"foo\": [true, \"bar\"], \"tags\": {\"a\": 1, \"b\": null} }";
		// Get the first index of a JSON array
		//query_string = "SELECT params->foo->0 FROM events";  // should return 'true'
		
		// for building json strings, generally and in particular for inserting new Tool confirgurations,
		// we can convert the contents of a Store into a json with the Store's streamer operator:
		Store astore; astore.Set("anint",12); astore.Set("dubs",33.); astore.Set("mystring","carrots");
		std::string astore_json; astore >> astore_json;
		query_string = "INSERT INTO lappd (created, configfiles, description) VALUES ( $1, $2, $3 )";
		// run it
		std::cout<<"testing writing Store json data with query: \n"<<query_string<<std::endl;
		myTransaction.exec_params(query_string,
		                         timestamp,
		                         astore_json,
		                         description
		                         );
		//myTransaction.commit();
		
		// handling binary data
		// ---------------------
		// to insert binary data we again use pqxx::binarystring, this time building it from
		// a pointer to the underlying data and its size
		// (n.b. the pointed-to data must be contiguous, so e.g. objects shouldn't contain pointers to data)
		// XXX NOTE: this includes std::containers such as XXX std::string XXX, std::vector etc, along with
		// anything else that internally holds such objects (such as Stores). char arrays are fine.
		std::cout<<"building a struct to test binary data\n";
		struct mystruct{
			int anint; double adub; bool abool; char sss[255]; //std::string ss; // << don't use std::string...!
			mystruct() : anint(12), adub(22.3), sss{'o','h','\0'}, abool(true) {}
			void Print(){
				std::cout<<"anint="<<anint<<", adub="<<adub<<", abool="<<abool<<", acharr="<<sss<<std::endl;
			}
		} a_struct;
		// XXX when constructing a pqxx::binarystring a copy of the underlying data is made XXX
		std::cout<<"converting to binarystring"<<std::endl;
		pqxx::binarystring binarystringdata(&a_struct, sizeof(mystruct));
		
		// to insert this into a query we may be able to use pqxx::transaction::esc_raw(mybinarystring)
		// but again it is MUCH safer to use a parameterized query. This separates the query arguments
		// so that we're not trying to embed arbitrary binary data into a string constant.
		// https://libpqxx.readthedocs.io/en/6.4/a01259.html#a08a78072ec1feeea9e86e034ca936fa3
		query_string = "INSERT INTO configfiles(system, name, version, description, created, author, contents)"
		               "VALUES ($1, $2, $3, $4, $5, $6, $7)";  // order of $N defines order of arguments
		std::cout<<"testing inserting binary data with parametrized query: \n"<<query_string<<"\n";
		version=4;
		results = myTransaction.exec_params( query_string,
		                                     systemname,
		                                     toolname,
		                                     version,
		                                     binarystringdata,
		                                     timestamp,
		                                     author,
		                                     astore_json );
		//myTransaction.commit();
		
		// for retrieval we again build a pqxx::binarystring from the pqxx::field,
		// then access the raw underlying binary via pqxx::binarystring.data() to obtain
		// a pointer to the data, and pqxx::binarystring.size() to know it's size in bytes.
		query_string = "SELECT description FROM configfiles WHERE name = "+myTransaction.quote(toolname)
		             + " AND system = " + myTransaction.quote(systemname) + " AND version = "
		             + pqxx::to_string(version);
		std::cout<<"testing reading binary data with query: \n"<<query_string<<"\n";
		results = myTransaction.exec(query_string);
		byteafield = results[0][0];
		my_bytea_fielddata = pqxx::binarystring(byteafield);
		mystruct* b_struct = (mystruct*)my_bytea_fielddata.data();
		
		std::cout<<"Input was: ";
		a_struct.Print();
		std::cout<<"Output was: ";
		b_struct->Print();
		
		// testing storing multi-line strings into text fields - it works just fine.
		query_string = "INSERT INTO configfiles(system, name, version, description, created, author, contents)"
		               "VALUES ($1, $2, $3, $4, $5, $6, $7)";  // order of $N defines order of arguments
		version=5;
		toolname="mutliple.\n Lines.\n Woah.\n";
		std::cout<<"storing configfiles entry with tool name: \n"<<toolname<<std::endl;
		results = myTransaction.exec_params( query_string,
		                                     systemname,
		                                     toolname,
		                                     version,
		                                     binarystringdata,
		                                     timestamp,
		                                     author,
		                                     astore_json );
		
		// check we can get it back out
		query_string = "SELECT name FROM configfiles WHERE version = 5";
		results = myTransaction.exec(query_string);
		std::cout<<"queried name was "<<results[0][0].as<string>()<<std::endl;
		
		std::cout<<"ok done, ending"<<std::endl;
	}catch (const pqxx::sql_error &e){
		// pqxx has its own exceptions, and in particular we can get information about
		// what was being attmpted when the exception was thrown:
		std::cerr << e.what() << std::endl
		          << "When executing query: " << e.query() << std::endl;
		return false;
	}
	// for good practice we should also handle any other thrown exceptions, just in case.
	catch (std::exception const &e){
		std::cerr << e.what() << std::endl;
		return false;
	};
	
	std::cout<<"execute ending"<<std::endl;
	
	return true;
}


bool PGTool::Finalise(){
  std::cout<<"PGTool finalising"<<std::endl;
  return true;
}



















































//	// useful for converting rows to a Store....?
//	// https://rudra.dev/posts/generate-beautiful-json-from-postgresql/
//	// could this be used for sending returned data over the network?
//	query_string = "SELECT row_to_json(row('id', 'name')) FROM user";
//	/*
//	  +-------------------------+
//	  | row_to_json             |
//	  |-------------------------|
//	  | {"id":1,"name":"user1"} |
//	  | {"id":2,"name":"user2"} |
//	  +-------------------------+
//	*/
//	// The above examples return us multiple JSON objects(one for each row)
//	
//	// if we want to combine multiple returned rows we can use aggregate functions
//	// https://newbedev.com/postgresql-return-result-set-as-json-array
//	// (https://www.postgresql.org/docs/9.5/functions-aggregate.html)
//	// https://dba.stackexchange.com/questions/69655/select-columns-inside-json-agg
//	query_string = "SELECT array_to_json(array_agg(row_to_json(users))) FROM ( "
//	             +     "SELECT id, name FROM user"
//	             + " ) users";
//	// OR
//	query_string = "SELECT json_agg(row_to_json(users)) FROM ( "
//	             + "FROM ( SELECT id, name FROM user"
//	             + " ) users";
//	// OR
//	query_string = "SELECT json_agg(user) FROM user";  // entire table selected
//	/*
//	  +----------------------------------------------------+
//	  | json_agg                                           |
//	  |----------------------------------------------------|
//	  | [{"id":1,"name":"user1"}, {"id":2,"name":"user2"}] |
//	  +----------------------------------------------------+
//	*/
//	// The above queries aggregate all the JSON objects into an array using 'array_agg',
//	// then convert them to a JSON array by applying array_to_json function.
//	// We could also have used 'json_agg' instead of array_agg, which would result in each row
//	// as an object, instead of JSON string...maybe...
//	
//	// https://stackoverflow.com/questions/27215216/postgres-how-to-convert-a-json-string-to-text
//	// https://stackoverflow.com/questions/49527710/postgresql-jsonb-string-format
