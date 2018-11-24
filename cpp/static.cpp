/*
Simple example where static int is declared inside a function and increment in each call.
By definition, static int variable declared inside a function retains its value between calls.
*/
#include <iostream>
#include <string.h>
using namespace std;

int getStatic() {
    static int counter = 1;
    int counter1 = 1;
    counter *= 2; 
    return counter;
}
int main() {
    int i = 0;
    
    for (i = 0; i < 10; i ++) {
        cout << getStatic() << endl;
    }

    return 0;
}
