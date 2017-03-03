#include <iostream>
#include <stdlib.h>
#include <ctime>
#include <algorithm>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/foreach.hpp>
#include <boost/program_options.hpp>
//#include <boost/filesystem.hpp>

namespace{
	const size_t ERROR_IN_COMMAND_LINE = 1;
	const size_t SUCCESS = 0;
	const size_t ERROR_UNHANDLED_EXCEPTION = 2;

}

int getParametersBoost(int argc, char **argv, int *debug, std::vector<std::string> *fileInVector){
	namespace po = boost::program_options;
	po::options_description desc("Options");
	desc.add_options()
		("debug", po::value<int>(debug)->required(), "debug options")
		("fileIn", po::value<std::vector<std::string>>(fileInVector), "a list of input files"); //--fileIn <value1> --fileIn <value2> --fileIn <value3>
	po::variables_map vm;
	try{
		po::store(po::parse_command_line(argc, argv, desc), vm);
		po::notify(vm);
	}
	catch (po::error& e){
		std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
		std::cerr << desc << std::endl;
		return ERROR_IN_COMMAND_LINE;
	}
}


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv){
	std::cout << " ============================================================================== " << std::endl;
	std::cout << " ------- from CPP read JSON   28/02/2017 -  David Valenzuela Urrutia ---------- " << std::endl;
	std::cout << " ============================================================================== " << std::endl;
	
	////////////////////////////////////////////////////////////////////////////////////
	// PARAMETROS DE ENTRADA
	////////////////////////////////////////////////////////////////////////////////////
	std::vector<std::string> fileInVector;
	int debug = 0; // default

	getParametersBoost(argc, argv, &debug, &fileInVector);

	std::cout << "debug = " << debug << std::endl;

	int fileInVectorSize = int(fileInVector.size());
	for (int i = 0; i < fileInVectorSize; i++){
		std::cout << "fileInVector (" << i << " de " << fileInVectorSize-1 << ") = " << fileInVector[i] << std::endl;
	}

	std::string fileIn;
	if (fileInVectorSize > 1){
		fileIn = fileInVector[0];
	}

	////////////////////////////////////////////////////////////////////////////////////
	// READ JSON PARAMETERS
	////////////////////////////////////////////////////////////////////////////////////
	using boost::property_tree::ptree;

	std::ifstream jsonFile(fileIn);
	ptree ptParam;
	read_json(jsonFile, ptParam);


	BOOST_FOREACH(boost::property_tree::ptree::value_type &v, ptParam.get_child("parametroStr1")) {
		std::cout << v.second.data() << std::endl;
	}

	BOOST_FOREACH(boost::property_tree::ptree::value_type &v, ptParam.get_child("Ymax")) {
		std::cout << v.first.data() << std::endl;
	}


	BOOST_FOREACH(boost::property_tree::ptree::value_type &v, ptParam.get_child("Xunique")) {
		std::cout << v.second.data() << std::endl;
	}

	float Ymax = ptParam.get<float>("Ymax");
	std::cout << Ymax << std::endl;

	std::string parametroStr2 = ptParam.get<std::string>("parametroStr2");
	std::cout << parametroStr2 << std::endl;
		
	std::cout << "DONE." << std::endl;
	return 0;
}