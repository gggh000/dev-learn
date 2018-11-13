/*
Simple polymorphism example. The Carp and Tuna is derived class from base class Fish and tuna is instantiated 
as fish1. From main fish1.swim() invokes Tuna::swim() and passed onto copyFish function its reference.
The copyFish function calls also Tuna::swim() thansk for Swim() declared as virtual in base class.
If you erase the virtual keyword from base class Swim() function, it is no longer poliymorphism and Fish::swim()
instead will be called. 
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

    virtual void Swim()
    {
        if (DEBUG == 1)
            cout << "Fish::Swim() entered." << endl;
    }
};

class Carp : public Fish
{
public:
    Carp()
    {
        if (DEBUG == 1)
            cout << "Carp() constructor entered." << endl;
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

int main()
{
    Tuna fish1;
    fish1.Swim();
    copyFish(fish1);
    return 0;
}
