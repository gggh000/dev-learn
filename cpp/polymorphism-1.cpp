/*
Pure virtual function example. The Swim() method of base Fish class is declared as pure virtual
function, there by rendering the Fish to be uninstantiable. 
Because of this, if line: //    Fish myFish; 
is commented out, the compiler error will result.
*/
#include <iostream>
using namespace std;
static int DEBUG = 1;

class Fish
{
public:
    Fish()
    {
        if (DEBUG == 1) 
            cout << "Fish() constructor entered." << endl;
    }

    virtual ~Fish()
    {
        if (DEBUG == 1)
            cout << "~Fish() destructor entered." << endl;
    }

    virtual void Swim() = 0;
    /*
    {
        if (DEBUG == 1)
            cout << "Fish::Swim() entered." << endl;
    }
    */
};

class Carp : public Fish
{
public:
    Carp()
    {
        if (DEBUG == 1)
            cout << "Carp() constructor entered." << endl;
    }

    ~Carp()
    {
        if (DEBUG == 1) 
            cout << "~Carp() destructor called." << endl;
    }

    void Swim()
    {
        if (DEBUG == 1)
            cout << "Carp::Swim() entered." << endl;
    }
};

class Tuna : public Fish
{
public:
    Tuna()
    {
        if (DEBUG == 1) 
            cout << "Tuna() constructor entered." << endl;
    }

    ~Tuna()
    {
        if (DEBUG == 1)
            cout << "~Tuna() destructor called." << endl;
    }

    void Swim()
    {
        if (DEBUG == 1)
            cout << "Tuna::Swim() entered." << endl;
    }
};

void copyFish(Fish & pFish)
{
    pFish.Swim();
}

void makeFishSwim(Fish & inputFish) 
{
    inputFish.Swim();
}

void deleteFishMemory(Fish * pFish)
{
    delete pFish;
}
int main()
{
//    Fish myFish;
    Carp myLunch;
    Tuna myDinner;

    makeFishSwim(myLunch);
    makeFishSwim(myDinner);
}
