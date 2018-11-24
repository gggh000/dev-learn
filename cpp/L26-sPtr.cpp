#include <iostream>
using namespace std;

class sPtr
{
    int * data;

public:

    // Constructor. 

    sPtr (int pData = NULL) {
        cout << "sPtr constructor." << endl;
        data = new int(pData);
    }

    // Destructor.

    ~sPtr () { 
        cout << "sPtr desctructor." << endl;
        delete data; 
    }

    // Copy assignment constructor.
    
    sPtr & operator = (sPtr & p) {
        cout << "sPtr copy assignment operator. " << endl;
        data = new int(p.getData());
        delete &p;
    }

    // Copy constructor.
    
    sPtr(sPtr & p) {
        cout << "sPtr copy constructor. " << endl;
        data = new int(p.getData());
    }

    // Accessory.
        
    int getData() { return * data; } 

    // Print info.

    void printInfo() {
        cout << "object addr:   " << hex << this << endl;
        cout << "data ptr addr: " << hex << data << endl;

        if (data != NULL) {
            cout << "data value:    " << dec << *data << endl; 
        } else { 
            cout << "data value is NULL." << endl;
        }
    }

private:
    
};

int main()
{
    sPtr int1(100);
    sPtr int2 ( int1 );
    //cout << "int1 / 2 ptr: " << &int1 << " / " << &int2 << endl; 

    cout << "---------------------" << endl;
    int1.printInfo();
    cout << "---------------------" << endl;
    int2.printInfo();
    cout << "---------------------" << endl;
    sPtr int3 = int1;
    int3.printInfo();
    cout << "---------------------" << endl;
    return 0;
}
