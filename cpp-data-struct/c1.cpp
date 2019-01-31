#include <iostream>
#include "c1.h"
using namespace std;

// default contructor.

c1::c1 ( void ) {
    cout << endl << "class c1: default constructor is called.";
}

// copy constructor.

c1::c1 ( const c1 &  pC1 ) {
    cout << endl << "class c1: copy constructor is called with: " << pC1.value;
    value  = pC1.value;
}

// Init value constructor.

c1::c1 ( int pInt1) {
    cout << endl << "class c1: constructor with init value is called with: " <<  pInt1;
    value = pInt1;
}

// move constructor.

c1::c1 (c1&& pC1 ) {
    cout << endl << "class c1: move constructor is called with: " <<  pC1.value;
    value = pC1.value;
    pC1.value = 0;
}

// copy assign constructor.

c1 c1::operator= (const c1 & pC1) {

    cout << endl << "class c1: copy assignment constructor is called with: " << pC1.value;

    if (&pC1 != this) { 
        c1 * localC1;
        localC1 = new c1 ( pC1.value );
        return * localC1;
    } else {
        cout << endl << "class c1: copy assignment aborted, assigning to same object.";
    }
}

// move assignment contructor.

c1 c1::operator= (c1 && pC1) {
    cout << endl << "class c1: move assignment constructor is called with: " << pC1.value;

    c1 localC1;
    std::swap(value, pC1.value);
    return localC1;
}

c1 function1 ( c1 pC1 ) {
    cout << endl << " function1 entered...";
    cout << endl << " function1 exiting...";

    c1 * lC1;
    lC1 = new c1(pC1);
    return *lC1;
}

c1 function2 ( c1 pC1 ) {
    cout << endl << " function1 entered...";
    cout << endl << " function1 exiting...";

    return pC1;
}



