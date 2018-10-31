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
