/*Simple thread app which launches hello and waits.
Use: g++ -std=gnu++0x -pthread hello.cpp to build.
*/

#include <iostream>
#include <thread>

void hello()
{
	std::cout << "Hello CONCURRENT WORLD\n"; 
}

int main()
{
	std::thread t(hello);
	t.join();
}
