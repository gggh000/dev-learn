/*
This program exemplifies the use of global variable, firstNumber and secondNumber declared
outside the main(). Once it is declared outside main() function, it is global and can be
used both inside main and multiplyNumber functions. 
*/
#include <iostream>
using namespace std;

// three global integers;

int firstNumber = 0;
int secondNumber = 0;
int multiplicationResult = 0;

void multiplyNumbers()
{
    cout << "Enter the 1st No.: ";
    cin >> firstNumber;

    cout << "Enter the second No,: ";
    cin >> secondNumber;
    
    multiplicationResult = firstNumber * secondNumber;

    cout << "Displaying from Multiplenumbers(): ";
    cout << firstNumber << " x " << secondNumber;
    cout << " = " << multiplicationResult << endl;
}

int main()
{
    cout << "This program will help you multiply two numbers" << endl;
    
    // Call the function that does all the work

    multiplyNumbers();

    cout << "Displaying from main(); ";
    
    // This line will not compile and work;

    cout << firstNumber << " x " << secondNumber;
    cout << " = " << multiplicationResult << endl;
}
