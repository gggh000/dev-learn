#include <iostream>
using namespace std;

class sPtr
{
    int * data;

public:

    // constructor. 

    sPtr (int pData = NULL) {
        cout << "sPtr constructor." << endl;
        data = new int(pData);
    }

    // destructor.

    ~sPtr () { 
        cout << "sPtr desctructor." << endl;
        delete data; 
    }

    // copy assignment constructor.
    
    sPtr & operator = (sPtr & p) {
        cout << "sPtr copy assignment operator. " << endl;
        sPtr * copy = new sPtr(p.getData());
        delete &p;
        return *copy; 
    }

    // copy constructor.
    
    sPtr(sPtr & p) {
        cout << "sPtr copy constructor. " << endl;
        data = new int(p.getData());
    }

    // accessory.
        
    int getData() { return *data; } 

    // print info

    void printInfo()
    {
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
    sPtr * int1;    
    int1 = new sPtr(100);
    sPtr * int2 ( int1 );
    sPtr * int3;
    int3 = int1;
    cout << "int1 / 2 ptr: " << int1 << " / " << int2 << endl; 

    cout << "---------------------" << endl;
    int1->printInfo();
    cout << "---------------------" << endl;
    int2->printInfo();
    cout << "---------------------" << endl;
    int3->printInfo();
    cout << "---------------------" << endl;
    delete int1;
    return 0;
}
