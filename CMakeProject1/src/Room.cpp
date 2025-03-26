#include "Room.h"
#include <filesystem>

Room::Room()
{
	for (auto& num : cost_map)
		num = 0;

	if (!tilemap.loadFromFile("data/tilemap_packed.png"))
	{
		std::cout << "Tilemap failed to load\n";
	}
}

bool Room::init(std::string path, bool build_map)
{
	if (!image.loadFromFile(path))
	{
		std::cout << "Failed to load " + path + "\n";
		return false;
	}

	for (int i = 0; i < HEIGHT; i++)
	{
		for (int j = 0; j < WIDTH; j++)
		{

			switch (int(image.getPixel(j, i).r))
			{
			case 0:
			case 1:
			case 2:
			case 3:
			case 4:
			case 5:
			case 6:
			case 7:
			case 8:
			case 9:
			{
				cost_map[i * WIDTH + j] = 1;
				break;
			}
			default:
			{
				cost_map[i * WIDTH + j] = 0;
				break;
			}
			}

			std::cout << int(image.getPixel(j, i).r) << " " << int(image.getPixel(j, i).g) << " " << int(image.getPixel(j, i).b) << " | ";
		}
		std::cout << std::endl;
	}

	if (build_map)
		buildMap();

	return true;
}

void Room::buildMap()
{
	for (int i = 0; i < HEIGHT; i++)
	{
		for (int j = 0; j < WIDTH; j++)
		{
			if (cost_map[i * WIDTH + j])
				layer.emplace_back(std::make_unique<Tile>(tilemap, j, i, 6, 1, SCALE));
			else
				layer.emplace_back(std::make_unique<Tile>(tilemap, j, i, 14, 0, SCALE));
		}
	}
}

void Room::update(float dt)
{

}

void Room::render(sf::RenderWindow& window)
{
	for (auto& tile : layer)
	{
		tile->render(window);
	}
}

sf::Vector2i Room::world(std::tuple<int, int> tile_coords)
{
	return sf::Vector2i{ (std::get<0>(tile_coords) + 1) * 16 - (16 / 2),
						(std::get<1>(tile_coords) + 1) * 16 - (16 / 2) };
}

std::tuple<int, int> Room::tile(sf::Vector2f point)
{
	return { int(point.x / 16),
			 int(point.y / 16) };
}
