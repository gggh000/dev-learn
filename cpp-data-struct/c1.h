#include <iostream>
using namespace std;

class c1
{
public:
    int value;

    c1 ( const c1 &  pC1 );
    c1 ( int pInt1);
    c1 (c1&& pC1 );
};

c1 function1 ( c1 pC1 );
c1 function2 ( c1 pC1 );

