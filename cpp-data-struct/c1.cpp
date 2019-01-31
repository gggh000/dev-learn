#include <iostream>
#include "c1.h"
using namespace std;

c1::c1 ( const c1 &  pC1 ) {
    cout << endl << " class c1: copy constructor is called with: " << pC1.value;
    value  = pC1.value;
}

c1::c1 ( int pInt1) {
    cout << endl << " class c1: constructor with init value is called with: " <<  pInt1;
    value = pInt1;
}

c1::c1 (c1&& pC1 ) {
    cout << endl << " class c1: move constructor is called with: pC1.value";
    value = pC1.value;
    pC1.value = 0;
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



