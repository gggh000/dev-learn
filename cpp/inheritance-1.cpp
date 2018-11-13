/*Simple example of private inheritance 
Uncommenting //myCar.fireCylinders(); line in main() will cause compilation error
because it is inherited as private.
*/

#include <iostream>
using namespace std;

class Motor
{
public:
    void switchIgnition()
    { cout << "Ignition on." << endl; }

    void pumpFuel()
    { cout << "Fuel in cylinders." << endl; }

    void fireCylinders()
    { cout << "Engine started." << endl; }
};

class Car : private Motor 
{
public:
    void move()
    {
        switchIgnition();
        pumpFuel();
        fireCylinders();
    }
};

/*
class RaceCar : protected Car
{
public:
    void move()
    {
        switchIgnition();
        pumpFuel();
        fireCylinders();
        fireCylinders();
        fireCylinders();
        fireCylinders();
    }
};
*/

int main()
{
    Car myCar;
    myCar.move();
    //myCar.fireCylinders();
    //RaceCar myCar1;
    //myCar1.move();
    return 0;
}
