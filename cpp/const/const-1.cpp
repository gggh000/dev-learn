#include <iostream>
#include <string>

using namespace std;

const int f1a () {
    const int i = 100;

    cout << "f1a: returning 100..." << endl;
    return i;
}
int f1 ( const int & j) {
    cout << "f1: j: " << j << endl;

    /* Following is not allowed since passed as const.
    const-1.cpp:18:7: error: assignment of read-only reference ‘j’
    */

    //j = 10;
    return j;
}

int f1b ( int const & j) {
    cout << "f1b: j: " << j << endl;
    /* Following is not allowed since passed as const.
    Uncomment to see following compile error:
    const-1.cpp:25:7: error: assignment of read-only reference ‘j’
     j = 10;

    */
    //j = 10;

    return j;
}

/*  non-class member function can not have const qualifier 
Uncomment to see compile error: 
const-1.cpp:15:18: error: non-member function ‘int f2(int)’ cannot have cv-qualifier
 int f2 ( int k ) const {
*/

/*
int f2 ( int k ) const {
    k = 10;
    cout << "f2: k: " << k << endl;
}
*/

class   cF2 {
    int m;

public:
    cF2 () {
        m = 10;
    }

    int f2 (  ) const {
        /* Following assignment is not allowed since the member function f2 is declared as const.
        Uncomment to see following compile error:
        const-1.cpp:37:11: error: assignment of member ‘cF2::m’ in read-only object
        */ 
        //m = 20;
        cout << "cF2.f2: m: " << m << endl;
    }
};


int main() {
    const int i = 0; 
    int p;

    cF2 cf2;
    cf2.f2();

    // Following is not allowed since declared as const int i;
    // i = 1;

    cout << "const int i: " << i << endl;    

    p = f1a ();    
    cout << "f1a: return: " << p << endl;
    p = 200;
    cout << "f1a: return modded: " << p << endl;

    cout <<  "Done... " << endl;

    f1 ( 200 );
    f1b ( 250 );

    return 0;

}
