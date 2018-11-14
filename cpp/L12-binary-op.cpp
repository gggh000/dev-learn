/*
Operator overloading examples for unary operator.
1. ++ prefix
2. postfix ++ 
3. operator const char*() implements the << operator. ostringstream formattedDate is used
by feeding the integers. And resulting string in formattedDate is assigned to private variable
dateInString because formattedDate is local to a function: operator const char*() and will
be destroyed when function returns.
*/

#include <iostream>
#include <sstream>
#include <string>

using namespace std;

class Date
{
private:
    int day, month, year;
    string dateInString;

public:
    Date (int inMonth, int inDay, int inYear)
        : month(inMonth), day(inDay), year(inYear) {};

    // Unary operator implementations ++, --, const char*.

    // prefix increment.

    Date & operator ++ ()
    {
        ++day;
        return *this;
    }

    // prefix decrement

    Date & operator -- ()
    {
        --day;
        return *this;
    }

    operator const char*()
    {
        ostringstream formattedDate;
        formattedDate << month << " / " << day << " / " << year;

        dateInString = formattedDate.str();
        return dateInString.c_str();
    }

    // Binary operator implementations.

    // Binary date addition / subtraction as well ass addition and subtraction assignment.

    Date operator + (int daysToAdd)
    {
        Date newDate(month, day + daysToAdd, year);
        return newDate;
    }    
    
    Date operator - (int daysToSub)
    {
        Date newDate(month, day - daysToSub, year);
        return newDate;
    }

    void operator+= (int daysToAdd)
    {
        day += daysToAdd;
    }

    void operator-= (int daysToSub)
    {
        day -= daysToSub;
    }

    bool operator==(Date & cmp)
    {
        if ((month == cmp.month) && (day == cmp.day) && (year == cmp.year))
            return true;
        else
            return false;
    }

    bool operator != (Date & cmp)
    {
        return !(this->operator==(cmp));
    }
    
    void displayDate()
    { 
        cout << month << " / " << day << " / " << year << endl; 
    }
};

int main()
{
    Date holiday(12, 25, 2016);
    cout << "The date object is initialized to: ";
    holiday.displayDate();
    
    cout << "---------UNARY OPERATOR OVERLOAD EXAMPLES---------" << endl;    

    ++holiday;
    cout << "Date after prefix-increment is: ";
    holiday.displayDate();

    --holiday;
    cout << "Date after a prefix-decrement is: ";
    holiday.displayDate();

    cout << "Print using cout operator << :";
    cout << holiday << endl;

    string strHoliday (holiday);
    strHoliday = Date(11, 11, 2016);

    cout << "Printing holiday: ";
    cout << strHoliday << endl;

    cout << "---------BINARY OPERATOR OVERLOAD EXAMPLES---------" << endl;    

    Date PreviousHoliday(holiday - 19);
    cout << "Previous holiday on: " << endl;
    PreviousHoliday.displayDate();

    Date NextHoliday(holiday + 6);
    cout << "Next holiday on: " << endl;
    NextHoliday.displayDate();

    cout << "holiday -= 19 gives: ";
    holiday -= 19;
    holiday.displayDate();

    cout << "holiday ++ 25 gives: ";
    holiday += 25;
    holiday.displayDate();

    Date holiday1 (12, 25, 2016);
    Date holiday2 (12, 25, 2011);
    
    cout << "holiday 1 is: ";
    holiday1.displayDate();

    cout << "holiday 2 is: ";
    holiday2.displayDate();
    
    if (holiday1 == holiday2) 
        cout << "holiday1 and 2 is equal using == operator." << endl;
    else
        cout << "holiday1 and 2 is NOT equal using == operator." << endl;

    if (holiday1 != holiday2) 
        cout << "holiday1 and 2 is NOT equal using != operator." << endl;
    else
        cout << "holiday1 and 2 is equal using != operator." << endl;
        

    // Implementation of smart pointer. 
    
    /*
    unique_ptr<int> smartIntPtr(new int);
    *smartIntPtr = 42;

    cout << "Integer value pointed by smart pointer is: " << *smartIntPtr << endl;

    unique_ptr<Date> smartHoliday (new Date(12, 25, 2016));
    cout << "The new instance of date contains: ";

    smartHoliday->displayDate();
    */
    
    return 0;
}



