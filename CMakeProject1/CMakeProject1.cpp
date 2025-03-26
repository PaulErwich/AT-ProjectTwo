// CMakeProject1.cpp : Defines the entry point for the application.
//

#include "CMakeProject1.h"
#include <SFML/Graphics.hpp>

#include "src/Room.h"
#include "src/Pathfinding.h"
#include "src/RoomValidater.h"

using namespace std;

int main()
{
	std::string filename = "generateImage.py";
	std::string command = "python ";
	command += filename;
	int filesToGen = 5;
	command += " " + std::to_string(filesToGen);
	system(command.c_str());


	sf::RenderWindow window(sf::VideoMode(1280, 720), "Dungeon Game");
	window.setFramerateLimit(60);

	Room room;

	//room.init("data/genOutput/samples/0005.png");

	bool test = canPathfind(room, Location(0, 14), Location(12, 31));

	RoomValidator temp;

	temp.validateRoom("data/genOutput/samples/0005.png");

	std::cout << test;

	sf::Clock clock;

	while (window.isOpen())
	{
		sf::Event event;

		sf::Time time = clock.restart();
		float dt = time.asSeconds();

		while (window.pollEvent(event))
		{
			if (event.type == sf::Event::Closed)
			{
				window.close();
			}
		}

		window.clear(sf::Color::Black);

		if (window.isOpen())
		{
			room.render(window);
			window.display();
		}
	}

	cout << "Hello CMake." << endl;
	return 0;
}
