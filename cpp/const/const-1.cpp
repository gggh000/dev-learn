#include <iostream>
#include <string>

using namespace std;

int f1 ( const int & j) {
    cout << "f1: j: " << j << endl;

    // Following is not allowed since passed as const.
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

    cF2 cf2;
    cf2.f2();

    // Following is not allowed since declared as const int i;
    // i = 1;

    cout << "const int i: " << i << endl;    
    cout <<  "Done... " << endl;
    return 0;

}
