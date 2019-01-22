#include <iostream>
#include <memory>
#include <utility>

using namespace std;
 
int main() {
    shared_ptr<int> valuePtr (new int (222) );
    cout << "valuePtr: " << *valuePtr << endl;
    shared_ptr<int> valuePtrNow(valuePtr);
    cout << "valuePtrNow: " << *valuePtrNow << endl;

    /*
    if (valuePtr != NULL)
        printf("0x%x", valuePtr);
    else    
        printf("valuePtr is null now.")

    if (valuePtrNow != NULL)
        printf("0x%x", valuePtrNow);
    else
        printf("valuePtrNow is null now.")
    */
    return 0;
}

