// CMakeProject1.cpp : Defines the entry point for the application.
//

#include "CMakeProject1.h"
#include <SFML/Graphics.hpp>
#include <filesystem>

#include "src/Room.h"
#include "src/Pathfinding.h"
#include "src/RoomValidater.h"
#include "src/Dungeon.h"

using namespace std;

int main()
{
	srand((unsigned)time(0));

	std::string filename = "generateImage.py";
	std::string command = "python ";
	command += filename;
	int filesToGen = 50;
	command += " " + std::to_string(filesToGen);

	sf::RenderWindow window(sf::VideoMode(1280, 1280), "Dungeon Game");
	window.setFramerateLimit(60);

	RoomValidator temp;

	Dungeon test;
	test.GenerateDungeonLayout();
	test.BuildFullDungeon(true);


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

			if (event.type == sf::Event::KeyPressed)
			{
				if (event.key.code == sf::Keyboard::G)
				{
					system(command.c_str());
				}
				if (event.key.code == sf::Keyboard::V)
				{
					auto dirIter = std::filesystem::directory_iterator("data/genOutput/samples/");
					for (auto& entry : dirIter)
					{
						//std::cout << entry.path() << std::endl;
						//std::cout << entry.path().filename() << std::endl;
						if (entry.is_regular_file())
						{
							temp.validateRoom(entry.path().string());
						}
					}
				}
				if (event.key.code == sf::Keyboard::B)
				{
					// Build dungeon code
					test.BuildFullDungeon();
				}
			}
		}

		window.clear(sf::Color::Black);

		if (window.isOpen())
		{
			test.render(window);

			window.display();
		}
	}

	cout << "Hello CMake." << endl;
	return 0;
}
