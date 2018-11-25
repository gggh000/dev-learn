/*Simple thread app which launches hello and waits.
Use: g++ -std=gnu++0x -pthread hello.cpp to build.
*/

#include <iostream>
#include <thread>
#include <chrono>
#include <unistd.h>
#include <exception>

using namespace std;

void hello(int pId, int pStat[])
{
    int sleep_duration = rand() % 10 + 20;
	std::cout << pId << ": Hello CONCURRENT WORLD, sleeping for " << sleep_duration << endl; 
    sleep(sleep_duration);
    pStat[pId]  = 1;
    std::cout << pId << ": done sleeping exiting now...";
}

int main()
{
    int i;
    const int CONFIG_THREAD_COUNT = 10;
    int stat[CONFIG_THREAD_COUNT];
    int sum;

    for ( i = 0; i < CONFIG_THREAD_COUNT; i ++ ) {
        std::thread t(hello, i, stat);
        t.detach();
    }

    sleep(50);
    cout << "Checking thread status-s..." << endl;

    /*
     while (sum != 10)  {
        sum = 0;

        for (i = 0; i < CONFIG_THREAD_COUNT; i++) {
            sum += stat[i];
        }

        sleep(5);
        cout << "main(): sum: " << sum << ". waiting for all threads to finish..." << endl;
    }
    */
}
