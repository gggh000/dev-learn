#include <iostream>
#include <math.h>
#include <TutorialConfig.h>
#include <MathFunctions.h>

using namespace std;
int main(int argc, char *argv[]) {
   if (argc < 2) {

        // report version.

        std::cout << argv[0] << " Version " << Tutorial_VERSION_MAJOR << "."
                            << Tutorial_VERSION_MINOR << std::endl;
        std::cout << "Usage: " << argv[0] << " number" << std::endl;
        return 1;
    }

    const double inputValue = std::stod(argv[1]);
    cout << "Using math lib sqrt" << endl;
    const double outputValue = sqrt(inputValue);
    cout << "output: " << outputValue << endl;
    return 0;
}
