#include <iostream>
#include <math.h>
//#include <cmath.h>

using namespace std;

double mysqrt(double x) { 
    
    #if defined(HAVE_LOG) && defined(HAVE_EXP)
      double result = exp(log(x) * 0.5);
      std::cout << "Computing sqrt of " << x << " to be " << result
                << " using log and exp" << std::endl;
    #else
      double result = x;
    #endif

    return result;
        
}
