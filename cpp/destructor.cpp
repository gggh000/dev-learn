#include <iostream>
#include <string>
using namespace std;

class myString
{
private:
    char * buffer;

public:
    myString(const char * initString)
    {
        if(initString != NULL) 
        {
            buffer = new char [strlen(initString) + 1];
            strcpy(buffer, initString);
        }
        else
            buffer = NULL;
    }    
};

int main()
{
}
