#include <iostream>
using namespace std;

class c1
{
public:
    int value;

    // default constructor.

    c1 ( void );

    // copy constructor.

    c1 ( const c1 &  pC1 );

    // Init value constructor.

    c1 ( int pInt1);

    // move constructor.

    c1 (c1&& pC1 );

    // copy assign constructor.

    c1 operator= (const c1 & pC1);

    // move assignment contructor.

    c1 operator= (c1 && pC1);

};

c1 function1 ( c1 pC1 );
c1 function2 ( c1 pC1 );

