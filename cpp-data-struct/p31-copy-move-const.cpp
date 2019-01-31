#include <iostream>

using namespace std;

class c1
{
public:
    int value;

    c1 ( const c1 &  pC1 ) {
        cout << endl << "- class c1: copy constructor is called.";
        value  = pC1.value;
    }

    c1 ( int pInt1) {
        cout << endl << "- class c1: constructor with init value is called.";
        value = pInt1;
    }

    c1 (c1&& pC1 ) {
        cout << endl << "- class c1: move constructor is called.";
        value = pC1.value;
        pC1.value = 0;
    }
};

int main ()
{
    /*
    This examples illustrates the usage of copy and move constructor and explicitly demonstrates in which instances";
    copy and move constructor are invoked in regards to a object in question:

    - declaration with initialization i.e.

    <type> B = C;       // copy construc if C is lvalue, move construct if C is rvalue.
    <type> B { C };     // copy construc if C is lvalue, move construct if C is rvalue.

    but not 
    B = C;              // assignment operator 

    - an object passed using call-by-vaue (instead of & or const &), which, as mentioned earlier, should rarely by done anyway.

    - an object returned by value (instead of by & or const & ). Again, copy constructor is invoked if the object being return is 
    lvalue and move constructor is invoked if it is rvalue.

    */

    cout << "1. Copy constructor using declaration with initialization, c2 is created with copy-constructor using c1.";
    
    cout << endl << "c1 being created with 100 init value.";
    c1 c1_inst ( 100 );
    cout << endl << "c2 being created with c1_inst as init value.";
    c1 c2_inst ( c1_inst );
    
    cout << endl;    
    
    return 0;
}

