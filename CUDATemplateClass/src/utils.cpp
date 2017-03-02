#include "utils.h"

void coutDeb(int debug, std::string stringOut){
	if (debug >= 1){
		std::cout << stringOut << std::endl;
	}
}