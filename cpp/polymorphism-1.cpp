/*
Simple polymorphism example. The Carp and Tuna is derived class from base class Fish and tuna is instantiated 
as fish1. From main fish1.swim() invokes Tuna::swim() and passed onto copyFish function its reference.
The copyFish function calls also Tuna::swim() thansk for Swim() declared as virtual in base class.
If you erase the virtual keyword from base class Swim() function, it is no longer poliymorphism and Fish::swim()
instead will be called. 

Finally virtual destructor "virtual ~Fish()" ensures that the derived class destructor be called through
polymorphism feature when pointer to fish is created and memory allocated and then finally be destroyed through
DeleteFishMemory.
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
    Carp myLunch;
    Tuna myDinner;

    makeFishSwim(myLunch);
    makeFishSwim(myDinner);
}
