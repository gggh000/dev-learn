#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int main()
{
    vector <int> intArray;
    
    intArray.push_back(50);
    intArray.push_back(51);
    intArray.push_back(2991);
    intArray.push_back(53);

    cout << "The contents of the vector are: " << endl;
    vector <int>::iterator arrIterator = intArray.begin();

    while (arrIterator != intArray.end())
    {
        cout << *arrIterator << endl;
        ++ arrIterator;
    }

    vector <int>::iterator elFound = find (intArray.begin(), intArray.end(), 2991);

    if (elFound != intArray.end())
    {
        int elPos = distance (intArray.begin(), elFound);
        cout << "Value " << *elFound;
        cout << " found in the vector at position: " << elPos << endl;
    } else { 
        cout << "Not found." << endl; 
    }
    
}
