#include <iostream>
using namespace std;
#include <string.h>

class MyString
{
private:
    char * buffer;

public:
    MyString(const char * initialInput)
    {
        if (initialInput != NULL) 
        {
            buffer = new char [strlen(initialInput) + 1];
            strcpy(buffer, initialInput);
        }
        else
            buffer = NULL;
    }

    MyString & operator=(const MyString & copySource)
    {
        if ((this != &copySource) && copySource.buffer != NULL)
        {
            if (buffer != NULL)
                delete[] buffer;

            buffer = new char [strlen(copySource.buffer) + 1];
            strcpy(buffer, copySource.buffer);
        }
        return *this;
    }   

    operator const char*()
    {
        return buffer;
    }

    ~MyString()
    {
        delete[] buffer;
    }    
};

int main()
{
    MyString string1("Hello");
    MyString string2(" World");

    cout << "Before assignment: " << endl;
    cout << string1 << string2 << endl;

    string2 = string1;

    cout << "After assignment: " << endl;
    cout << string1 << string2 << endl;
}


