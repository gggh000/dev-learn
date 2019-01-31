#include <iostream>
#include "c1.h"
using namespace std;

int main ()
{
    /*
    THIS COMMENT IS WRONG!!!!
    This examples illustrates the usage of copy and move constructor and explicitly demonstrates in which instances";
    copy and move constructor are invoked in regards to a object in question:

    1. declaration with initialization i.e.

    <type> B = C;       // copy construc if C is lvalue, move construct if C is rvalue.
    <type> B { C };     // copy construc if C is lvalue, move construct if C is rvalue.

    but not 
    B = C;              // assignment operator 

    2. an object passed using call-by-vaue (instead of & or const &), which, as mentioned earlier, should rarely by done anyway.

    3. an object returned by value (instead of by & or const & ). Again, copy constructor is invoked if the object being return is 
    lvalue and move constructor is invoked if it is rvalue.

    */

    cout << endl << "1. Invoking copy assignment constructor.";

    cout << endl << "Creating c1_inst with init value: 100 ";

    c1 c1_inst ( 100 );

    cout << endl << "Creating c6_inst with default constructor.";

    c1 c6_inst;

    cout << endl << "Assign c1_inst to c6_inst";

    c6_inst =  c1_inst;

    cout << endl << "Create c7_inst with move assignment constructor.";

    c1 c7_inst;
    c7_inst = std::move(c1_inst);

    cout << endl;
    
    return 0;
}

